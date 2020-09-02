#include <string>

static const std::string init_time_str           = "Initialization time : ";
static const std::string preprocessing_time_str  = "Preprocessing time  : ";
static const std::string dedispersion_time_str   = "Dedispersion time   : ";
static const std::string postprocessing_time_str = "Postprocessing time : ";
static const std::string input_memcpy_time_str   = "Input memcpy time   : ";
static const std::string output_memcpy_time_str  = "Output memcpy time  : ";
static const std::string total_time_str          = "Total time          : ";

static const std::string preprocessing_perf_str  = "Preprocessing performance : ";
static const std::string dedispersion_perf_str   = "Dedispersion performance  : ";

static const std::string info_str             = ">> Info";
static const std::string debug_str            = ">> Debug";
static const std::string timings_str          = ">> Timings";
static const std::string memory_alloc_str     = ">> Allocate memory";
static const std::string fft_plan_str         = ">> Create FFT plan";
static const std::string prepare_input_str    = ">> Prepare input";
static const std::string fft_r2c_str          = ">> FFT input r2c";
static const std::string fft_c2r_str          = ">> FFT output c2r";
static const std::string ref_dedispersion_str = ">> Perform dedispersion in time domain";
static const std::string fdd_dedispersion_str = ">> Perform dedispersion in frequency domain";
static const std::string copy_output_str      = ">> Copy output";