/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* This file contains the C wrappers for the CPP library.
* And was adapted from the original dedisp.cu (Copyright 2012 Ben Barsdell).
*/

//#define DEDISP_DEBUG

#include "dedisp.h"
#include <memory> //for shared pointers
#include <iostream>
#include <typeinfo>

// Implementation specific CPP Plan interfaces
#include <dedisp/DedispPlan.hpp>
#include <tdd/TDDPlan.hpp>
#include <fdd/FDDGPUPlan.hpp>
#include <fdd/FDDCPUPlan.hpp>

#ifdef __cplusplus
extern "C" {
#endif

// Define plan structure
// with a C-compatible interface to a CPP dedisp::Plan
struct dedisp_plan_struct {
	// shared_ptr to implementation plan
	std::shared_ptr<dedisp::Plan> ptr = nullptr;
};

// Global device index
static int g_device_idx = 0;
// Global implementation selection,
// default to orignal dedispersion implemantation
static dedisp_implementation g_implementation = DEDISP_DEDISP;

// Internal abstraction for errors
#ifdef DEDISP_DEBUG
#define throw_error(error) do {                                         \
	printf("An error occurred within dedisp on line %d of %s: %s",      \
	       __LINE__, __FILE__, dedisp_get_error_string(error));         \
	return (error); } while(0)
#define throw_getter_error(error, retval) do {                          \
	printf("An error occurred within dedisp on line %d of %s: %s",      \
	       __LINE__, __FILE__, dedisp_get_error_string(error));         \
	return (retval); } while(0)
#else
#define throw_error(error) return error
#define throw_getter_error(error, retval) return retval
#endif // DEDISP_DEBUG

/*
dedisp_error throw_error(dedisp_error error) {
	// Note: Could, e.g., put an error callback in here
	return error;
}
*/

/* Currently not implemented in C-interface:
dedisp_error update_scrunch_list(dedisp_plan plan) {
	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}
	if( !plan->scrunching_enabled || 0 == plan->dm_count ) {
		plan->scrunch_list.resize(0);
		// Fill with 1's by default for safety
		plan->scrunch_list.resize(plan->dm_count, dedisp_size(1));
		return DEDISP_NO_ERROR;
	}
	plan->scrunch_list.resize(plan->dm_count);
	dedisp_error error = generate_scrunch_list(&plan->scrunch_list[0],
	                                           plan->dm_count,
	                                           plan->dt,
	                                           &plan->dm_list[0],
	                                           plan->nchans,
	                                           plan->f0,
	                                           plan->df,
	                                           plan->pulse_width,
	                                           plan->scrunch_tol);
	if( error != DEDISP_NO_ERROR ) {
		return error;
	}

	// Allocate on and copy to the device
	try {
		plan->d_scrunch_list.resize(plan->dm_count);
	}
	catch(...) {
		throw_error(DEDISP_MEM_ALLOC_FAILED);
	}
	try {
		plan->d_scrunch_list = plan->scrunch_list;
	}
	catch(...) {
		throw_error(DEDISP_MEM_COPY_FAILED);
	}

	return DEDISP_NO_ERROR;
}
// ------------------------
*/

// Public functions
// ----------------
dedisp_error dedisp_create_plan(dedisp_plan* plan,
                                dedisp_size  nchans,
                                dedisp_float dt,
                                dedisp_float f0,
                                dedisp_float df)
{
#ifdef DEDISP_DEBUG
	std::cout <<  "dedisp_create_plan()" << std::endl;
#endif

	// Initialise to NULL for safety
	*plan = 0;

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	// create plan structure
	*plan = new dedisp_plan_struct();
	if( !plan ) {
		throw_error(DEDISP_MEM_ALLOC_FAILED);
	}

	// Set environment variable USE_CPU to switch to CPU implementation of FDD
  	// Using GPU implementation by default
	char *use_cpu_str = getenv("USE_CPU");
  	bool use_cpu = !use_cpu_str ? false : atoi(use_cpu_str);

	try {
		// switch case based on optional implementation parameter
		switch (g_implementation)
		{
		case DEDISP_DEDISP:
			(*plan)->ptr.reset(new dedisp::DedispPlan(nchans, dt, f0, df, g_device_idx));
		break;

		case DEDISP_TDD:
			(*plan)->ptr.reset(new dedisp::TDDPlan(nchans, dt, f0, df, g_device_idx));
		break;

		case DEDISP_FDD:
  			if (use_cpu) (*plan)->ptr.reset(new dedisp::FDDCPUPlan(nchans, dt, f0, df, g_device_idx));
			else (*plan)->ptr.reset(new dedisp::FDDGPUPlan(nchans, dt, f0, df, g_device_idx));
		break;

		default:
			dedisp_destroy_plan(*plan);
			return DEDISP_UNKNOWN_ERROR; //ToDo: add new ERROR for unknown implementation ?
		}
	}
	catch(...) {
		dedisp_destroy_plan(*plan);
		throw_error(DEDISP_UNKNOWN_ERROR); //ToDo: add new ERROR for fail on plan creation ?
	}

	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_set_gulp_size(dedisp_plan plan,
                                  dedisp_size gulp_size)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	try
	{
		// Only DedispPlan and TDDPlan have a set gulp size interface
		if(typeid(*plan->ptr.get())==typeid(dedisp::DedispPlan))
		{
			static_cast<dedisp::DedispPlan*>(plan->ptr.get())->set_gulp_size(gulp_size);
		}
		else if(typeid(*plan->ptr.get())==typeid(dedisp::TDDPlan))
		{
			static_cast<dedisp::TDDPlan*>(plan->ptr.get())->set_gulp_size(gulp_size);
		}
		else if(typeid(*plan->ptr.get())==typeid(dedisp::FDDGPUPlan)) throw_error(DEDISP_INVALID_PLAN);
		else if(typeid(*plan->ptr.get())==typeid(dedisp::FDDCPUPlan)) throw_error(DEDISP_INVALID_PLAN);
		else throw_error(DEDISP_UNKNOWN_ERROR);
	}
	catch(...)
	{
		throw_error(DEDISP_UNKNOWN_ERROR);
	}

	return DEDISP_NO_ERROR;
}

dedisp_size dedisp_get_gulp_size(dedisp_plan plan)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	try
	{
		// Only DedispPlan has a set gulp size interface
		if(typeid(*plan->ptr.get())==typeid(dedisp::DedispPlan))
		{
			return static_cast<dedisp::DedispPlan*>(plan->ptr.get())->get_gulp_size();
		}
		else if(typeid(*plan->ptr.get())==typeid(dedisp::TDDPlan))
		{
			return static_cast<dedisp::TDDPlan*>(plan->ptr.get())->get_gulp_size();
		}
		else if(typeid(*plan->ptr.get())==typeid(dedisp::FDDGPUPlan)) throw_getter_error(DEDISP_INVALID_PLAN,0);
		else if(typeid(*plan->ptr.get())==typeid(dedisp::FDDCPUPlan)) throw_getter_error(DEDISP_INVALID_PLAN,0);
		else throw_getter_error(DEDISP_UNKNOWN_ERROR,0);
	}
	catch(...)
	{
		throw_getter_error(DEDISP_UNKNOWN_ERROR,0);
	}

}

dedisp_error dedisp_set_dm_list(dedisp_plan plan,
                                const dedisp_float* dm_list,
                                dedisp_size count)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	try
	{
		plan->ptr->set_dm_list(dm_list, count);
	}
	catch(...)
	{
		throw_error(DEDISP_UNKNOWN_ERROR);
	}

	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_generate_dm_list(dedisp_plan plan,
                                     dedisp_float dm_start, dedisp_float dm_end,
                                     dedisp_float ti, dedisp_float tol)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	try
	{
		plan->ptr->generate_dm_list(dm_start, dm_end, ti, tol);
	}
	catch(...)
	{
		throw_error(DEDISP_UNKNOWN_ERROR);
		// ToDo: error passing from Plan methods to here might be improved
	}

	return DEDISP_NO_ERROR;
}

/* Currently not implemented in C-interface:
dedisp_float * dedisp_generate_dm_list_guru (dedisp_float dm_start, dedisp_float dm_end,
            double dt, double ti, double f0, double df,
            dedisp_size nchans, double tol, dedisp_size * dm_count)
{
  std::vector<dedisp_float> dm_table;
  generate_dm_list(dm_table,
           dm_start, dm_end,
           dt, ti, f0, df,
           nchans, tol);
  *dm_count = dm_table.size();
  return &dm_table[0];
}
*/

dedisp_error dedisp_set_device(int device_idx) {
	//keep global copy of device_idx for usage in dedisp_create_plan()
	g_device_idx = device_idx;

	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_select_implementation(dedisp_implementation imp) {
	//keep global copy of selected implementation for usage in dedisp_create_plan()
	g_implementation = imp;

	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_set_killmask(dedisp_plan plan, const dedisp_bool* killmask)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	try
	{
		plan->ptr->set_killmask(killmask);
	}
	catch(...)
	{
		throw_error(DEDISP_UNKNOWN_ERROR);
		// ToDo: error passing from Plan methods to here might be improved
	}

	return DEDISP_NO_ERROR;
}

// Getters
// -------
dedisp_size         dedisp_get_max_delay(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	if( 0 == plan->ptr->get_dm_count()) { throw_getter_error(DEDISP_NO_DM_LIST_SET,0); }
	return plan->ptr->get_max_delay();
}

dedisp_size         dedisp_get_dm_delay(const dedisp_plan plan, int dm_trial) {
  if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
  if( 0 == plan->ptr->get_dm_count() ) { throw_getter_error(DEDISP_NO_DM_LIST_SET,0); }
  if (dm_trial < 0 || (dedisp_size)dm_trial >= plan->ptr->get_dm_count() ) { throw_getter_error(DEDISP_UNKNOWN_ERROR,0); }
  // ToDo: make dm_trial of type dedisp_size such that it can not be smaller than 0, then cast to int to index dm_list
  // changing this has impact on the interface of the function.
  return (plan->ptr->get_dm_list()[dm_trial] * plan->ptr->get_delay_table()[plan->ptr->get_channel_count()-1] + 0.5);
}

dedisp_size         dedisp_get_channel_count(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->ptr->get_channel_count();
}

dedisp_size         dedisp_get_dm_count(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->ptr->get_dm_count();
}

const dedisp_float* dedisp_get_dm_list(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	if( 0 == plan->ptr->get_dm_count() ) { throw_getter_error(DEDISP_NO_DM_LIST_SET,0); }
	return &plan->ptr->get_dm_list()[0];
}

const dedisp_bool*  dedisp_get_killmask(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return &plan->ptr->get_killmask()[0];
}

dedisp_float        dedisp_get_dt(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->ptr->get_dt();
}

dedisp_float        dedisp_get_f0(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->ptr->get_f0();
}

dedisp_float        dedisp_get_df(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->ptr->get_df();
}

// Execution functions
// -------
dedisp_error dedisp_execute_guru(const dedisp_plan  plan,
                                 dedisp_size        nsamps,
                                 const dedisp_byte* in,
                                 dedisp_size        in_nbits,
                                 dedisp_size        in_stride,
                                 dedisp_byte*       out,
                                 dedisp_size        out_nbits,
                                 dedisp_size        out_stride,
                                 dedisp_size        first_dm_idx,
                                 dedisp_size        dm_count,
                                 unsigned           flags)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	try
	{
		/* Dedisp and TDD have a guru interface, check type.
		*  Cast to specific Plan because the Dedisp interface has flags and
		*  the TDD interface does not have flags.*/
		if(typeid(*plan->ptr.get())==typeid(dedisp::DedispPlan))
		{
#ifdef DEDISP_DEBUG
			std::cout << "dedisp_execute_guru() dedisp::DedispPlan with flags" << std::endl;
#endif
			static_cast<dedisp::DedispPlan*>(plan->ptr.get())->execute_guru(	nsamps,
																			in,
																			in_nbits,
																			in_stride,
																			out,
																			out_nbits,
																			out_stride,
																			first_dm_idx,
																			dm_count,
																			flags);
		}
		else if(typeid(*plan->ptr.get())==typeid(dedisp::TDDPlan))
		{
#ifdef DEDISP_DEBUG
			std::cout << "dedisp_execute_guru() dedisp::TDDPlan without flags" << std::endl;
#endif
			static_cast<dedisp::TDDPlan*>(plan->ptr.get())->execute_guru(	nsamps,
																			in,
																			in_nbits,
																			in_stride,
																			out,
																			out_nbits,
																			out_stride,
																			first_dm_idx,
																			dm_count);
		}
		else if(typeid(*plan->ptr.get())==typeid(dedisp::FDDGPUPlan)) throw_error(DEDISP_INVALID_PLAN);
		else if(typeid(*plan->ptr.get())==typeid(dedisp::FDDCPUPlan)) throw_error(DEDISP_INVALID_PLAN);
		else throw_error(DEDISP_UNKNOWN_ERROR);
	}
	catch(...)
	{
		throw_error(DEDISP_UNKNOWN_ERROR);
	}

	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_execute_adv(const dedisp_plan  plan,
                                dedisp_size        nsamps,
                                const dedisp_byte* in,
                                dedisp_size        in_nbits,
                                dedisp_size        in_stride,
                                dedisp_byte*       out,
                                dedisp_size        out_nbits,
                                dedisp_size        out_stride,
                                unsigned           flags)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	try
	{
		/* Dedisp and TDD have an advanced interface, check type.
		*  Cast to specific Plan because the Dedisp interface has flags and
		*  the TDD interface does not have flags.*/
		if(typeid(*plan->ptr.get())==typeid(dedisp::DedispPlan))
		{
#ifdef DEDISP_DEBUG
			std::cout << "dedisp_execute_adv() dedisp::DedispPlan with flags" << std::endl;
#endif
			static_cast<dedisp::DedispPlan*>(plan->ptr.get())->execute_adv(	nsamps,
																			in,
																			in_nbits,
																			in_stride,
																			out,
																			out_nbits,
																			out_stride,
																			flags);
		}
		else if(typeid(*plan->ptr.get())==typeid(dedisp::TDDPlan))
		{
#ifdef DEDISP_DEBUG
			std::cout << "dedisp_execute_adv() dedisp::TDDPlan without flags" << std::endl;
#endif
			static_cast<dedisp::TDDPlan*>(plan->ptr.get())->execute_adv(	nsamps,
																			in,
																			in_nbits,
																			in_stride,
																			out,
																			out_nbits,
																			out_stride);
		}
		else if(typeid(*plan->ptr.get())==typeid(dedisp::FDDGPUPlan)) throw_error(DEDISP_INVALID_PLAN);
		else if(typeid(*plan->ptr.get())==typeid(dedisp::FDDCPUPlan)) throw_error(DEDISP_INVALID_PLAN);
		else throw_error(DEDISP_UNKNOWN_ERROR);
	}
	catch(...)
	{
		throw_error(DEDISP_UNKNOWN_ERROR);
	}

	return DEDISP_NO_ERROR;
}

// TODO: Consider having the user specify nsamps_computed instead of nsamps
dedisp_error dedisp_execute(const dedisp_plan  plan,
                            dedisp_size        nsamps,
                            const dedisp_byte* in,
                            dedisp_size        in_nbits,
                            dedisp_byte*       out,
                            dedisp_size        out_nbits,
                            unsigned           flags)
{

	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }

	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	try
	{
		plan->ptr->execute(nsamps, in, in_nbits, out, out_nbits, flags);
	}
	catch(...)
	{
		throw_error(DEDISP_UNKNOWN_ERROR);
	}

	return DEDISP_NO_ERROR;
}

// ToDo: duplicate of dedisp::Plan.sync()
// would be better to change interface to use the function from Plan class
dedisp_error dedisp_sync(void)
{
	try
	{
		cu::checkError(cudaDeviceSynchronize());
	}
	catch(...)
	{
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	return DEDISP_NO_ERROR;
}

void dedisp_destroy_plan(dedisp_plan plan)
{
	if( plan ) {
		delete plan;
	}
}

const char* dedisp_get_error_string(dedisp_error error)
{
	switch( error ) {
	case DEDISP_NO_ERROR:
		return "No error";
	case DEDISP_MEM_ALLOC_FAILED:
		return "Memory allocation failed";
	case DEDISP_MEM_COPY_FAILED:
		return "Memory copy failed";
	case DEDISP_INVALID_DEVICE_INDEX:
		return "Invalid device index";
	case DEDISP_DEVICE_ALREADY_SET:
		return "Device is already set and cannot be changed";
	case DEDISP_NCHANS_EXCEEDS_LIMIT:
		return "No. channels exceeds internal limit";
	case DEDISP_INVALID_PLAN:
		return "Invalid plan";
	case DEDISP_INVALID_POINTER:
		return "Invalid pointer";
	case DEDISP_INVALID_STRIDE:
		return "Invalid stride";
	case DEDISP_NO_DM_LIST_SET:
		return "No DM list has been set";
	case DEDISP_TOO_FEW_NSAMPS:
		return "No. samples < maximum delay";
	case DEDISP_INVALID_FLAG_COMBINATION:
		return "Invalid flag combination";
	case DEDISP_UNSUPPORTED_IN_NBITS:
		return "Unsupported in_nbits value";
	case DEDISP_UNSUPPORTED_OUT_NBITS:
		return "Unsupported out_nbits value";
	case DEDISP_PRIOR_GPU_ERROR:
		return "Prior GPU error.";
	case DEDISP_INTERNAL_GPU_ERROR:
		return "Internal GPU error. Please contact the author(s).";
	case DEDISP_UNKNOWN_ERROR:
		return "Unknown error. Please contact the author(s).";
	default:
		return "Invalid error code";
	}
}

/* Currently not implemented in C-interface:
dedisp_error dedisp_enable_adaptive_dt(dedisp_plan  plan,
                                       dedisp_float pulse_width,
                                       dedisp_float tol)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	plan->scrunching_enabled = true;
	plan->pulse_width = pulse_width;
	plan->scrunch_tol = tol;
	return update_scrunch_list(plan);
}
dedisp_error dedisp_disable_adaptive_dt(dedisp_plan plan) {
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	plan->scrunching_enabled = false;
	return update_scrunch_list(plan);
}
dedisp_bool dedisp_using_adaptive_dt(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,false); }
	return plan->scrunching_enabled;
}
const dedisp_size* dedisp_get_dt_factors(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	if( 0 == plan->dm_count ) { throw_getter_error(DEDISP_NO_DM_LIST_SET,0); }
	return &plan->scrunch_list[0];
}
*/

#ifdef __cplusplus
}
#endif
// ----------------
