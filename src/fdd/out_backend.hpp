// Output backends for FDDGPUPlan::execute_gpu.
//
// Backend selection is two-level:
//   * compile time: ENABLE_POSIX_AIO / ENABLE_IOURING decide which async
//     backends are built in (io_uring pulls in liburing); the synchronous
//     pwrite() backend is always built;
//   * run time:     make_out_backend() picks one of the compiled-in backends
//     (env DEDISP_OUT_BACKEND=sync|aio|uring; unset or unavailable picks the
//     best one built: io_uring, then POSIX AIO, then sync).
//
// Each backend is a concrete class (SyncBackend, PosixAioBackend,
// UringBackend) implementing a small virtual interface (OutBackend). The
// virtual call happens once per DM row write, next to a syscall, so its cost
// is irrelevant.
//
// NOTE: this header expects maybe_group_unlock() and kLustreGroupLockGid to
// be declared before it is included (they live in FDDGPUPlan.cpp).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <unistd.h>

#ifdef ENABLE_POSIX_AIO
#include <aio.h>
#endif
#ifdef ENABLE_IOURING
#include <liburing.h>
#endif

namespace dedisp {

// The part of a DM job the output backend needs to know about: which slot of
// the paged `out` buffer it reads from, and how many writes issued from that
// slot are still outstanding. DMData derives from this.
struct OutSlotJob {
  int slot = -1;
  std::atomic<unsigned> pending{0};
};

enum class OutBackendKind { Sync, PosixAIO, IoUring };

// ---------------------------------------------------------------------------
// Common interface + recycled output-slot pool (backend independent).
// ---------------------------------------------------------------------------
class OutBackend {
public:
  // The paged `out` buffer holds num_slots slots of ndm_batch_max rows each,
  // recycled as the writes that consume them complete. A DM job takes a slot
  // when the output thread is about to fill it, and gives it back once the
  // last write issued from it has been reaped.
  explicit OutBackend(unsigned num_slots) {
    for (unsigned s = 0; s < num_slots; ++s) free_slots_.push_back(s);
  }
  virtual ~OutBackend() = default;

  OutBackend(const OutBackend&) = delete;
  OutBackend& operator=(const OutBackend&) = delete;

  int acquire_slot() {
    std::unique_lock<std::mutex> lk(slot_mutex_);
    slot_cv_.wait(lk, [&] { return !free_slots_.empty(); });
    int slot = free_slots_.front();
    free_slots_.pop_front();
    return slot;
  }

  void release_slot(int slot) {
    {
      std::lock_guard<std::mutex> lk(slot_mutex_);
      free_slots_.push_back(slot);
    }
    slot_cv_.notify_one();
  }

  // Drop one outstanding write of `job` without issuing it (e.g. open()
  // failed); releases the slot if that was the last one.
  void complete_one(OutSlotJob* job) {
    if (job && job->pending.fetch_sub(1) == 1) release_slot(job->slot);
  }

  // Post one nonblocking write. buf must stay valid until the write is
  // reaped -- guaranteed because job's slot is only recycled once
  // job->pending drops to zero. Arm job->pending before the first post.
  // If owns_fd, the backend closes fd after the write completes.
  virtual void post_write(OutSlotJob* job, int fd, bool owns_fd, off_t offset,
                          const void* buf, size_t nbytes,
                          bool group_locked = false) = 0;

  // Start the completion thread. Call once, from the producer thread.
  virtual void start() = 0;

  // Wait until every posted write has landed and stop the completion
  // thread. After this returns no fd passed with owns_fd=false is in use.
  virtual void finish() = 0;

  virtual const char* name() const = 0;

protected:
  // Bookkeeping shared by both reapers once a write has completed.
  static void on_write_done(int fd, bool owns_fd, bool group_locked) {
    if (group_locked) fsync(fd);
    maybe_group_unlock(fd, kLustreGroupLockGid, group_locked);
    if (owns_fd) close(fd);
  }

  static void check_result(const char* what, long res, size_t nbytes) {
    if (res < 0) {
      std::cerr << what << " write failed: " << strerror((int)-res) << std::endl;
    } else if ((size_t)res != nbytes) {
      std::cerr << "short " << what << " write: " << res << " of " << nbytes
                << " bytes" << std::endl;
    }
  }

private:
  std::mutex slot_mutex_;
  std::condition_variable slot_cv_;
  std::deque<int> free_slots_;
};

// ---------------------------------------------------------------------------
// Synchronous pwrite() -- always available, no extra libraries.
// ---------------------------------------------------------------------------
class SyncBackend final : public OutBackend {
public:
  SyncBackend(unsigned num_slots, size_t /*max_inflight*/)
      : OutBackend(num_slots) {}

  const char* name() const override { return "sync"; }

  // No completion thread and nothing ever in flight.
  void start() override {}
  void finish() override {}

  // Blocking write on the caller's thread; the write, post-write
  // bookkeeping (fsync/unlock/close) and slot accounting all finish before
  // this returns.
  void post_write(OutSlotJob* job, int fd, bool owns_fd, off_t offset,
                  const void* buf, size_t nbytes,
                  bool group_locked = false) override {
    // pwrite() may write fewer bytes than asked (signals, or Linux's
    // ~2 GiB per-call cap), so loop until done.
    const char* p = static_cast<const char*>(buf);
    size_t left = nbytes;
    while (left > 0) {
      ssize_t n = pwrite(fd, p, left, offset);
      if (n < 0) {
        if (errno == EINTR) continue;
        check_result("pwrite", -errno, nbytes);
        break;
      }
      if (n == 0) {  // no progress; avoid spinning forever
        check_result("pwrite", (long)(nbytes - left), nbytes);
        break;
      }
      p += n;
      offset += n;
      left -= (size_t)n;
    }

    on_write_done(fd, owns_fd, group_locked);
    complete_one(job);
  }
};

// ---------------------------------------------------------------------------
// POSIX AIO
// ---------------------------------------------------------------------------
#ifdef ENABLE_POSIX_AIO
class PosixAioBackend final : public OutBackend {
public:
  PosixAioBackend(unsigned num_slots, size_t max_inflight)
      : OutBackend(num_slots), max_inflight_(max_inflight) {}

  ~PosixAioBackend() override {
    if (reap_thread_.joinable()) finish();
  }

  const char* name() const override { return "posix_aio"; }

  void start() override { reap_thread_ = std::thread([this] { reap_loop(); }); }

  // aio_write() is issued under mutex_ so the entry is never visible to the
  // reaper before it has been submitted to the kernel. Blocks only while the
  // ring is full, never on a completion.
  void post_write(OutSlotJob* job, int fd, bool owns_fd, off_t offset,
                  const void* buf, size_t nbytes,
                  bool group_locked = false) override {
    std::unique_lock<std::mutex> lk(mutex_);
    space_cv_.wait(lk, [&] { return inflight_.size() < max_inflight_; });
    inflight_.emplace_back();
    PendingWrite& pw = inflight_.back();
    pw.fd = fd;
    pw.owns_fd = owns_fd;
    pw.group_locked = group_locked;
    pw.job = job;
    pw.cb.aio_fildes = fd;
    pw.cb.aio_offset = offset;
    pw.cb.aio_buf = const_cast<void*>(buf);
    pw.cb.aio_nbytes = nbytes;
    pw.cb.aio_sigevent.sigev_notify = SIGEV_NONE;
    if (aio_write(&pw.cb) != 0) {
      // Not submitted: complete it immediately on the reaper side.
      pw.submit_errno = errno;
      std::cerr << "aio_write failed: " << strerror(errno) << std::endl;
    }
    lk.unlock();
    items_cv_.notify_one();
  }

  // End-of-stream sentinel: everything queued ahead of it is reaped first,
  // so joining after posting it means all writes have landed.
  void finish() override {
    {
      std::unique_lock<std::mutex> lk(mutex_);
      space_cv_.wait(lk, [&] { return inflight_.size() < max_inflight_; });
      inflight_.emplace_back();
      inflight_.back().stop = true;
    }
    items_cv_.notify_one();
    reap_thread_.join();
  }

private:
  // WHY A DEQUE: aio_write() hands the kernel the address of this aiocb,
  // which must stay put until completion. std::deque never invalidates
  // references on push_back/pop_front, so as long as only emplace_back and
  // pop_front are used every aiocb keeps a stable address.
  struct PendingWrite {
    int fd = -1;
    bool owns_fd = true;
    bool group_locked = false;
    bool stop = false;
    int submit_errno = 0;
    OutSlotJob* job = nullptr;
    struct aiocb cb {};
  };

  // Single producer / single consumer. Only the reaper pop_front()s, so the
  // reference it holds while waiting stays valid as the producer appends.
  void reap_loop() {
    while (true) {
      std::unique_lock<std::mutex> lk(mutex_);
      items_cv_.wait(lk, [&] { return !inflight_.empty(); });
      PendingWrite& pw = inflight_.front();
      const bool stop = pw.stop;
      lk.unlock();

      if (stop) {
        lk.lock();
        inflight_.pop_front();
        return;
      }

      if (pw.submit_errno == 0) {
        const struct aiocb* cblist[1] = {&pw.cb};
        while (aio_error(&pw.cb) == EINPROGRESS) {
          aio_suspend(cblist, 1, nullptr);
        }
        const int err = aio_error(&pw.cb);
        ssize_t ret = aio_return(&pw.cb);
        check_result("aio", ret < 0 ? -err : ret, pw.cb.aio_nbytes);
      }
      on_write_done(pw.fd, pw.owns_fd, pw.group_locked);
      OutSlotJob* job = pw.job;

      lk.lock();
      inflight_.pop_front();
      lk.unlock();
      space_cv_.notify_one();

      complete_one(job);
    }
  }

  size_t max_inflight_;
  std::deque<PendingWrite> inflight_;
  std::mutex mutex_;
  std::condition_variable space_cv_;  // producer waits for room
  std::condition_variable items_cv_;  // reaper waits for work
  std::thread reap_thread_;
};
#endif  // ENABLE_POSIX_AIO

// ---------------------------------------------------------------------------
// io_uring
// ---------------------------------------------------------------------------
#ifdef ENABLE_IOURING
class UringBackend final : public OutBackend {
public:
  // queue_depth = max in-flight writes. In-flight is already bounded by the
  // slot pool (num_slots * ndm_batch_max rows == out_buf_rows), so with
  // queue_depth = out_buf_rows the SQ never has to block for long and the CQ
  // (2x entries by default) can never overflow.
  UringBackend(unsigned num_slots, unsigned queue_depth)
      : OutBackend(num_slots) {
    int ret = io_uring_queue_init(queue_depth, &ring_, 0);
    if (ret < 0) {
      throw std::runtime_error(std::string("io_uring_queue_init failed: ") +
                               strerror(-ret));
    }
  }

  ~UringBackend() override {
    if (reap_thread_.joinable()) finish();
    io_uring_queue_exit(&ring_);
  }

  const char* name() const override { return "io_uring"; }

  void start() override { reap_thread_ = std::thread([this] { reap_loop(); }); }

  // Only the producer thread touches the SQ and only the reaper touches the
  // CQ, which liburing supports without extra locking.
  void post_write(OutSlotJob* job, int fd, bool owns_fd, off_t offset,
                  const void* buf, size_t nbytes,
                  bool group_locked = false) override {
    io_uring_sqe* sqe = get_sqe();
    auto* req = new UringWrite{fd, owns_fd, group_locked, nbytes, job};
    io_uring_prep_write(sqe, fd, buf, nbytes, offset);
    io_uring_sqe_set_data(sqe, req);
    io_uring_submit(&ring_);
  }

  // Drain nop with null user data: IOSQE_IO_DRAIN makes it complete only
  // after every earlier SQE, so the reaper seeing it means all writes landed.
  void finish() override {
    io_uring_sqe* sqe = get_sqe();
    io_uring_prep_nop(sqe);
    io_uring_sqe_set_flags(sqe, IOSQE_IO_DRAIN);
    io_uring_sqe_set_data(sqe, nullptr);
    io_uring_submit(&ring_);
    reap_thread_.join();
  }

private:
  struct UringWrite {
    int fd;
    bool owns_fd;
    bool group_locked;
    size_t nbytes;
    OutSlotJob* job;
  };

  io_uring_sqe* get_sqe() {
    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
    while (!sqe) {
      io_uring_submit(&ring_);
      sqe = io_uring_get_sqe(&ring_);
    }
    return sqe;
  }

  void reap_loop() {
    while (true) {
      io_uring_cqe* cqe;
      int ret = io_uring_wait_cqe(&ring_, &cqe);
      if (ret == -EINTR) continue;
      if (ret < 0) {
        std::cerr << "io_uring_wait_cqe failed: " << strerror(-ret) << std::endl;
        return;
      }
      auto* req = static_cast<UringWrite*>(io_uring_cqe_get_data(cqe));
      const int res = cqe->res;
      io_uring_cqe_seen(&ring_, cqe);
      if (req == nullptr) return;  // drain sentinel

      check_result("io_uring", res, req->nbytes);
      on_write_done(req->fd, req->owns_fd, req->group_locked);
      OutSlotJob* job = req->job;
      delete req;
      complete_one(job);
    }
  }

  io_uring ring_{};
  std::thread reap_thread_;
};
#endif  // ENABLE_IOURING

// ---------------------------------------------------------------------------
// Runtime selection among the compiled-in backends.
// ---------------------------------------------------------------------------
inline OutBackendKind default_out_backend_kind() {
  const char* env = std::getenv("DEDISP_OUT_BACKEND");
  std::string v = env ? env : "";
#if defined(ENABLE_IOURING)
  if (v == "uring" || v == "io_uring") return OutBackendKind::IoUring;
#endif
#if defined(ENABLE_POSIX_AIO)
  if (v == "aio" || v == "posix_aio") return OutBackendKind::PosixAIO;
#endif
  if (v == "sync" || v == "pwrite") return OutBackendKind::Sync;
  if (!v.empty()) {
    std::cerr << "DEDISP_OUT_BACKEND=" << v
              << " not available in this build, using default" << std::endl;
  }
#if defined(ENABLE_IOURING)
  return OutBackendKind::IoUring;
#elif defined(ENABLE_POSIX_AIO)
  return OutBackendKind::PosixAIO;
#else
  return OutBackendKind::Sync;
#endif
}

inline std::unique_ptr<OutBackend> make_out_backend(OutBackendKind kind,
                                                    unsigned num_slots,
                                                    unsigned max_inflight) {
  switch (kind) {
#ifdef ENABLE_IOURING
    case OutBackendKind::IoUring:
      try {
        return std::make_unique<UringBackend>(num_slots, max_inflight);
      } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
#ifdef ENABLE_POSIX_AIO
        std::cerr << "falling back to POSIX AIO" << std::endl;
        return std::make_unique<PosixAioBackend>(num_slots, max_inflight);
#else
        std::cerr << "falling back to synchronous pwrite" << std::endl;
        return std::make_unique<SyncBackend>(num_slots, max_inflight);
#endif
      }
#endif
#ifdef ENABLE_POSIX_AIO
    case OutBackendKind::PosixAIO:
      return std::make_unique<PosixAioBackend>(num_slots, max_inflight);
#endif
    case OutBackendKind::Sync:
      return std::make_unique<SyncBackend>(num_slots, max_inflight);
    default:
      break;
  }
  throw std::runtime_error("requested output backend not compiled in");
}

}  // namespace dedisp
