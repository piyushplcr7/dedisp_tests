#include <string.h>
#include "barycenter_presto_utils.h"
#include <ctype.h>
#include <math.h>

FILE *chkfopen(char *path, const char *mode)
{
    FILE *file;

    if ((file = fopen(path, mode)) == NULL) {
        perror("\nError in chkfopen()");
        printf("   path = '%s'\n", path);
        exit(-1);
    }
    return (file);
}


size_t chkfread(void *data, size_t type, size_t number, FILE * stream)
{
    size_t num;

    num = fread(data, type, number, stream);
    if (num != number && ferror(stream)) {
        perror("\nError in chkfread()");
        printf("\n");
        exit(-1);
    }
    return num;
}


size_t chkfwrite(void *data, size_t type, size_t number, FILE * stream)
{
    size_t num;

    num = fwrite(data, type, number, stream);
    if (num != number && ferror(stream)) {
        perror("\nError in chkfwrite()");
        printf("\n");
        exit(-1);
    }
    return num;
}

double doppler(double freq_observed, double voverc)
/* This routine returns the frequency emitted by a pulsar */
/* (in MHz) given that we observe the pulsar at frequency */
/* freq_observed (MHz) while moving with radial velocity  */
/* (in units of v/c) of voverc wrt the pulsar.            */
{
    return freq_observed * (1.0 + voverc);
}


int read_resid_rec(FILE * file, double *toa, double *obsf)
/* This routine reads a single record (i.e. 1 TOA) from */
/* the file resid2.tmp which is written by TEMPO.       */
/* It returns 1 if successful, 0 if unsuccessful.       */
{
    static int firsttime = 1, use_ints = 0;
    static double d[9];

    // The default Fortran binary block marker has changed
    // several times in recent versions of g77 and gfortran.
    // g77 used 4 bytes, gfortran 4.0 and 4.1 used 8 bytes
    // and gfortrans 4.2 and higher use 4 bytes again.
    // So here we try to auto-detect what is going on.
    // The current version should be OK on 32- and 64-bit systems

    if (firsttime) {
        int ii;
        long long ll;
        double dd;

        chkfread(&ll, sizeof(long long), 1, file);
        chkfread(&dd, sizeof(double), 1, file);
        if (0)
            printf("(long long) index = %lld  (MJD = %17.10f)\n", ll, dd);
        if (ll != 72 || dd < 40000.0 || dd > 70000.0) { // 9 * doubles
            rewind(file);
            chkfread(&ii, sizeof(int), 1, file);
            chkfread(&dd, sizeof(double), 1, file);
            if (0)
                printf("(int) index = %d    (MJD = %17.10f)\n", ii, dd);
            if (ii == 72 && (dd > 40000.0 && dd < 70000.0)) {
                use_ints = 1;
            } else {
                fprintf(stderr,
                        "\nError:  Can't read the TEMPO residuals correctly!\n");
                exit(1);
            }
        }
        rewind(file);
        firsttime = 0;
    }
    if (use_ints) {
        int ii;
        chkfread(&ii, sizeof(int), 1, file);
    } else {
        long long ll;
        chkfread(&ll, sizeof(long long), 1, file);
    }
    //  Now read the rest of the binary record
    chkfread(&d, sizeof(double), 9, file);
    if (0) {                    // For debugging
        printf("Barycentric TOA = %17.10f\n", d[0]);
        printf("Postfit residual (pulse phase) = %g\n", d[1]);
        printf("Postfit residual (seconds) = %g\n", d[2]);
        printf("Orbital phase = %g\n", d[3]);
        printf("Barycentric Observing freq = %g\n", d[4]);
        printf("Weight of point in the fit = %g\n", d[5]);
        printf("Timing uncertainty = %g\n", d[6]);
        printf("Prefit residual (seconds) = %g\n", d[7]);
        printf("??? = %g\n\n", d[8]);
    }
    *toa = d[0];
    *obsf = d[4];
    if (use_ints) {
        int ii;
        return chkfread(&ii, sizeof(int), 1, file);
    } else {
        long long ll;
        return chkfread(&ll, sizeof(long long), 1, file);
    }
}

void barycenter(double *topotimes, double *barytimes,
                double *voverc, long N, char *ra, char *dec, char *obs, char *ephem)
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
{
    FILE *outfile;
    long i;
    double fobs = 1000.0, femit, dtmp;
    char command[100], temporaryfile[100];

    /* Make/chdir to a temp dir to avoid multiple prepfolds stepping on 
     * each other.
     */
    char tmpdir[]  = "/tmp/prestoXXXXXX";
    if (mkdtemp(tmpdir)==NULL) {
        fprintf(stderr, "barycenter: error creating temp dir.\n");
        exit(1);
    }
    char *origdir = getcwd(NULL,0);
    chdir(tmpdir);

    /* Write the free format TEMPO file to begin barycentering */

    strcpy(temporaryfile, "bary.tmp");
    outfile = chkfopen(temporaryfile, "w");
    fprintf(outfile, "C  Header Section\n"
            "  HEAD                    \n"
            "  PSR                 bary\n"
            "  NPRNT                  2\n"
            "  P0                   1.0 1\n"
            "  P1                   0.0\n"
            "  CLK            UTC(NIST)\n"
            "  PEPOCH           %19.13f\n"
            "  COORD              J2000\n"
            "  RA                    %s\n"
            "  DEC                   %s\n"
            "  DM                   0.0\n"
            "  EPHEM                 %s\n"
            "C  TOA Section (uses ITAO Format)\n"
            "C  First 8 columns must have + or -!\n"
            "  TOA\n", topotimes[0], ra, dec, ephem);

    /* Write the TOAs for infinite frequencies */

    for (i = 0; i < N; i++) {
        fprintf(outfile, "topocen+ %19.13f  0.00     0.0000  0.000000  %s\n",
                topotimes[i], obs);
    }
    fprintf(outfile, "topocen+ %19.13f  0.00     0.0000  0.000000  %s\n",
            topotimes[N - 1] + 10.0 / SECPERDAY, obs);
    fprintf(outfile, "topocen+ %19.13f  0.00     0.0000  0.000000  %s\n",
            topotimes[N - 1] + 20.0 / SECPERDAY, obs);
    fclose(outfile);

    /* Call TEMPO */

    /* Check the TEMPO *.tmp and *.lis files for errors when done. */

    sprintf(command, "tempo bary.tmp > tempoout_times.tmp");
    if (system(command) == -1) {
        fprintf(stderr, "\nError calling TEMPO in barycenter.c!\n");
        exit(1);
    }

    /* Now read the TEMPO results */

    strcpy(temporaryfile, "resid2.tmp");
    outfile = chkfopen(temporaryfile, "rb");

    /* Read the barycentric TOAs for infinite frequencies */

    for (i = 0; i < N; i++) {
        read_resid_rec(outfile, &barytimes[i], &dtmp);
    }
    fclose(outfile);

    /* rename("itoa.out", "itoa1.out"); */
    /* rename("bary.tmp", "bary1.tmp"); */
    /* rename("bary.par", "bary1.par"); */

    /* Write the free format TEMPO file to begin barycentering */

    strcpy(temporaryfile, "bary.tmp");
    outfile = chkfopen(temporaryfile, "w");
    fprintf(outfile, "C  Header Section\n"
            "  HEAD                    \n"
            "  PSR                 bary\n"
            "  NPRNT                  2\n"
            "  P0                   1.0 1\n"
            "  P1                   0.0\n"
            "  CLK            UTC(NIST)\n"
            "  PEPOCH           %19.13f\n"
            "  COORD              J2000\n"
            "  RA                    %s\n"
            "  DEC                   %s\n"
            "  DM                   0.0\n"
            "  EPHEM                 %s\n"
            "C  TOA Section (uses ITAO Format)\n"
            "C  First 8 columns must have + or -!\n"
            "  TOA\n", topotimes[0], ra, dec, ephem);

    /* Write the TOAs for finite frequencies */

    for (i = 0; i < N; i++) {
        fprintf(outfile, "topocen+ %19.13f  0.00  %9.4f  0.000000  %s\n",
                topotimes[i], fobs, obs);
    }
    fprintf(outfile, "topocen+ %19.13f  0.00  %9.4f  0.000000  %s\n",
            topotimes[N - 1] + 10.0 / SECPERDAY, fobs, obs);
    fprintf(outfile, "topocen+ %19.13f  0.00  %9.4f  0.000000  %s\n",
            topotimes[N - 1] + 20.0 / SECPERDAY, fobs, obs);
    fclose(outfile);

    /* Call TEMPO */

    /* Insure you check the file tempoout.tmp for  */
    /* errors from TEMPO when complete.            */

    sprintf(command, "tempo bary.tmp > tempoout_vels.tmp");
    if (system(command) == -1) {
        fprintf(stderr, "\nError calling TEMPO in barycenter.c!\n");
        exit(1);
    }

    /* Now read the TEMPO results */

    strcpy(temporaryfile, "resid2.tmp");
    outfile = chkfopen(temporaryfile, "rb");

    /* Determine the radial velocities using the emitted freq */

    for (i = 0; i < N; i++) {
        read_resid_rec(outfile, &dtmp, &femit);
        voverc[i] = femit / fobs - 1.0;
    }
    fclose(outfile);

    /* Cleanup the temp files */

    /* rename("itoa.out", "itoa2.out"); */
    /* rename("bary.tmp", "bary2.tmp"); */
    /* rename("bary.par", "bary2.par"); */

    remove("tempo.lis");
    remove("tempoout_times.tmp");
    remove("tempoout_vels.tmp");
    remove("resid2.tmp");
    remove("bary.tmp");
    remove("matrix.tmp");
    remove("bary.par");

    chdir(origdir);
    free(origdir);
    rmdir(tmpdir);
}

char *rmtrail(char *str)
/* Removes trailing space from a string */
{
    int i;

    if (str && 0 != (i = strlen(str))) {
        while (--i >= 0) {
            if (!isspace(str[i]))
                break;
        }
        str[++i] = '\0';
    }
    return str;
}

char *rmlead(char *str)
/* Removes leading space from a string */
{
    char *obuf;

    if (str) {
        for (obuf = str; *obuf && isspace(*obuf); ++obuf);
        if (str != obuf)
            strMove(str, obuf);
    }
    return str;
}

double dms2rad(int deg, int min, double sec)
/* Convert degrees, minutes, and seconds of arc to radians */
{
    double sign = 1.0;

    if (deg < 0)
        sign = -1.0;
    if (deg == 0 && (min < 0 || sec < 0.0))
        sign = -1.0;
    return sign * ARCSEC2RAD * (60.0 * (60.0 * (double) abs(deg)
                                        + (double) abs(min)) + fabs(sec));
}

double hms2rad(int hour, int min, double sec)
/* Convert hours, minutes, and seconds of arc to radians */
{
    return SEC2RAD * (60.0 * (60.0 * (double) hour + (double) min) + sec);
}

char *remove_whitespace(char *str)
/* Remove leading and trailing space from a string */
{
    return rmlead(rmtrail(str));
}

void ra_dec_from_string(char *radec, int *h_or_d, int *m, double *s)
/* Return a values for hours or degrees, minutes and seconds        */
/* given a properly formatted RA or DEC string.                     */
/*   radec is a string with J2000 RA  in the format 'hh:mm:ss.ssss' */
/*   or a string with J2000 DEC in the format 'dd:mm:ss.ssss'       */
{
    int retval;

    radec = remove_whitespace(radec);
    retval = sscanf(radec, "%d:%d:%lf\n", h_or_d, m, s);
    if (retval != 3) {
        char tmp[100];
        sprintf(tmp,
                "Error:  can not convert '%s' to RA or DEC in ra_dec_from_string()\n",
                radec);
        perror(tmp);
        exit(1);
    }
    if (radec[0] == '-' && *h_or_d == 0) {
        *m = -*m;
        *s = -*s;
    }
}

void hours2hms(double hours, int *h, int *m, double *s)
/* Convert decimal hours to hours, minutes, and seconds */
{
    double tmp;

    *h = (int) floor(hours);
    tmp = (hours - *h) * 60.0;
    *m = (int) floor(tmp);
    *s = (tmp - *m) * 60.0;
}

void deg2dms(double degrees, int *d, int *m, double *s)
/* Convert decimal degrees to degrees, minutes, and seconds */
{
    int sign = 1;
    double tmp;

    if (degrees < 0.0)
        sign = -1;
    *d = (int) floor(fabs(degrees));
    tmp = (fabs(degrees) - *d) * 60.0;
    *m = (int) floor(tmp);
    *s = (tmp - *m) * 60.0;
    *d *= sign;
    if (*d == 0) {
        *m *= sign;
        *s *= sign;
    }
}

void convertRAStrToDegrees(char ra_str[40], double* ra2000, char dec_str[40], double* dec2000) {
    int d, h, m;
    double sec;
    ra_dec_from_string(ra_str, &h, &m, &sec);
    *ra2000 = hms2rad(h, m, sec) * RADTODEG;

    ra_dec_from_string(dec_str, &d, &m, &sec);
    *dec2000 = dms2rad(d, m, sec) * RADTODEG;
}

void ra_dec_to_string(char *radec, int h_or_d, int m, double s)
/* Return a properly formatted string containing RA or DEC values   */
/*   radec is a string with J2000 RA  in the format 'hh:mm:ss.ssss' */
/*   or a string with J2000 DEC in the format 'dd:mm:ss.ssss'       */
{
    int offset = 0;

    if (h_or_d == 0 && (m < 0 || s < 0.0)) {
        radec[0] = '-';
        offset = 1;
    }
    sprintf(radec + offset, "%.2d:%.2d:%07.4f", h_or_d, abs(m), fabs(s));
}

void getTempoStrings(char ra_str_in[40], char dec_str_in[40], char ra_str_out[50], char dec_str_out[50]) {
    double ra2000, dec2000;
    convertRAStrToDegrees(ra_str_in, &ra2000, dec_str_in, &dec2000);

    int ra_h, ra_m;
    double ra_s;
    hours2hms(ra2000 / 15.0, &ra_h, &ra_m, &ra_s);

    int dec_d, dec_m;
    double dec_s;
    deg2dms(dec2000, &dec_d, &dec_m, &dec_s);

    ra_dec_to_string(ra_str_out, ra_h, ra_m, ra_s);
    ra_dec_to_string(dec_str_out, dec_d, dec_m, dec_s);
}

double getTloTOA(int mjd_i, double mjd_f) {
    return mjd_i + mjd_f;
}

long double getStartMJD(long double IMJD, long double SMJD, long double OFFS, long double BE_DELAY) {
    // This ignores the OFFS_SUB, INDEXVAL stuff in psrfits.c! 
    return IMJD + ((long double) SMJD + (long double) OFFS + (long double) BE_DELAY) / SECPERDAY;
}

char *strlower(char *str)
/* Convert a string to lower case */
{
    char *ss;

    if (str) {
        for (ss = str; *ss; ++ss)
            *ss = tolower(*ss);
    }
    return str;
}

void telescope_to_tempocode(char *inname, char *outname, char *obscode)
// Return the 2 character TEMPO string for an observatory
// whose name is in the string "inname".  Return a nice
// name in "outname".
{
    char scope[40];

    strncpy(scope, inname, 40);
    // ensure null-terminated
    scope[39] = '\0';
    strlower(scope);
    if (strcmp(scope, "gbt") == 0) {
        strcpy(obscode, "GB");
        strcpy(outname, "GBT");
    } else if (strcmp(scope, "arecibo") == 0) {
        strcpy(obscode, "AO");
        strcpy(outname, "Arecibo");
    } else if (strcmp(scope, "vla") == 0) {
        strcpy(obscode, "VL");
        strcpy(outname, "VLA");
    } else if (strcmp(scope, "parkes") == 0) {
        strcpy(obscode, "PK");
        strcpy(outname, "Parkes");
    } else if (strcmp(scope, "jodrell") == 0) {
        strcpy(obscode, "JB");
        strcpy(outname, "Jodrell Bank");
    } else if ((strcmp(scope, "gb43m") == 0) ||
               (strcmp(scope, "gb 140ft") == 0) || (strcmp(scope, "nrao20") == 0)) {
        strcpy(obscode, "G1");
        strcpy(outname, "GB43m");
    } else if (strcmp(scope, "nancay") == 0) {
        strcpy(obscode, "NC");
        strcpy(outname, "Nancay");
    } else if (strcmp(scope, "effelsberg") == 0) {
        strcpy(obscode, "EF");
        strcpy(outname, "Effelsberg");
    } else if (strcmp(scope, "srt") == 0) {
        strcpy(obscode, "SR");
        strcpy(outname, "Sardinia Radio Telescope");
    } else if (strcmp(scope, "fast") == 0) {
        strcpy(obscode, "FA");
        strcpy(outname, "FAST");
    } else if (strcmp(scope, "wsrt") == 0) {
        strcpy(obscode, "WT");
        strcpy(outname, "WSRT");
    } else if (strcmp(scope, "gmrt") == 0) {
        strcpy(obscode, "GM");
        strcpy(outname, "GMRT");
    } else if (strcmp(scope, "chime") == 0) {
        strcpy(obscode, "CH");
        strcpy(outname, "CHIME");
    } else if (strcmp(scope, "lofar") == 0) {
        strcpy(obscode, "LF");
        strcpy(outname, "LOFAR");
    } else if (strcmp(scope, "lwa") == 0) {
        strcpy(obscode, "LW");
        strcpy(outname, "LWA1");
    } else if (strcmp(scope, "mwa") == 0 ) {
        strcpy(obscode, "MW");
        strcpy(outname, "MWA");
    } else if (strcmp(scope, "meerkat") == 0 ) {
        strcpy(obscode, "MK");
        strcpy(outname, "MeerKAT");
    } else if (strcmp(scope, "ata") == 0) {
        strcpy(obscode, "AT");
        strcpy(outname, "ATA");
    } else if (strcmp(scope, "k7") == 0 ) {
        strcpy(obscode, "K7");
        strcpy(outname, "KAT-7");
    } else if (strcmp(scope, "geocenter") == 0) {
        strcpy(obscode, "0 ");
        strcpy(outname, "Geocenter");
    } else {
        printf("\nWARNING!!!:  I don't recognize the observatory (%s)!\n", inname);
        printf("                 Defaulting to the Geocenter for TEMPO.\n");
        strcpy(obscode, "0 ");
        strcpy(outname, "Unknown");
    }
}