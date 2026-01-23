#ifndef BARYCENTERPRESTOUTILSH
#define BARYCENTERPRESTOUTILSH

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#define SECPERDAY     86400.0
#define RADTODEG      57.29577951308232087679815481410517033240547246656
#define ARCSEC2RAD 4.8481368110953599358991410235794797595635330237270e-6
#define SEC2RAD    7.2722052166430399038487115353692196393452995355905e-5
#define TDT 1.0 //20.0

#define strMove(d,s) memmove(d,s,strlen(s)+1)

#define LININTERP(X, xlo, xhi, ylo, yhi) ((ylo)+((X)-(xlo))*((yhi)-(ylo))/((xhi)-(xlo)))

#ifdef __cplusplus
extern "C" {
#endif

FILE *chkfopen(char *path, const char *mode);


size_t chkfread(void *data, size_t type, size_t number, FILE * stream);


size_t chkfwrite(void *data, size_t type, size_t number, FILE * stream);

double doppler(double freq_observed, double voverc);


int read_resid_rec(FILE * file, double *toa, double *obsf);

void barycenter(double *topotimes, double *barytimes,
                double *voverc, long N, char *ra, char *dec, char *obs, char *ephem);
/* This routine uses TEMPO to correct a vector of           */
/* topocentric times (in *topotimes) to barycentric times   */
/* (in *barytimes) assuming an infinite observation         */
/* frequency.  The routine also returns values for the      */
/* radial velocity of the observation site (in units of     */
/* v/c) at the barycentric times.  All three vectors must   */
/* be initialized prior to calling.  The vector length for  */
/* all the vectors is 'N' points.  The RA and DEC (J2000)   */
/* of the observed object are passed as strings in the      */
/* following format: "hh:mm:ss.ssss" for RA and             */
/* "dd:mm:ss.ssss" for DEC.  The observatory site is passed */
/* as a 2 letter ITOA code.  This observatory code must be  */
/* found in obsys.dat (in the TEMPO paths).  The ephemeris  */
/* is the full name of an ephemeris supported by TEMPO,     */
/* examples include DE200, DE421, or DE436.                 */

char *rmtrail(char *str);

char *rmlead(char *str);

double dms2rad(int deg, int min, double sec);

double hms2rad(int hour, int min, double sec);

char *remove_whitespace(char *str);

void ra_dec_from_string(char *radec, int *h_or_d, int *m, double *s);

void hours2hms(double hours, int *h, int *m, double *s);

void deg2dms(double degrees, int *d, int *m, double *s);

void convertRAStrToDegrees(char ra_str[40], double* ra2000, char dec_str[40], double* dec2000);

void ra_dec_to_string(char *radec, int h_or_d, int m, double s);

void getTempoStrings(char ra_str_in[40], char dec_str_in[40], char ra_str_out[50], char dec_str_out[50]);

double getTloTOA(int mjd_i, double mjd_f);

long double getStartMJD(long double IMJD, long double SMJD, long double OFFS, long double BE_DELAY);

char *strlower(char *str);

void telescope_to_tempocode(char *inname, char *outname, char *obscode);

#ifdef __cplusplus
}
#endif

#endif

