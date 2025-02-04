Timing plan for FDD on GPU: variables: size of input, #DM trials (not controlled directly but can be)

* Reading data from disk and reducing it to a useful form

* Moving the data to GPU (cudamemcpy)

* dedispersion kernel on GPU (includes inverse fft as well? can be skipped)

* output time (can be circumvented if integrated pipeline like astro accelerate)

Ideas: Reading multiple fits files into one data array?

Comparing with presto dedispersion: keep input variables same and time similar components

