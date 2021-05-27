//////////////////////////////////////
// Cunren Liang, NASA JPL/Caltech
// Copyright 2017
//////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#define NR_END 1
#define FREE_ARG char*
#define PI 3.1415926535897932384626433832795028841971693993751058

typedef struct {
  float re;
  float im;
} fcomplex;


int *array1d_int(long nc){

  int *fv;

  fv = (int*) malloc(nc * sizeof(int));
  if(!fv){
    fprintf(stderr,"Error: cannot allocate 1-D int array\n");
    exit(1);
  }

  return fv;
}

void free_array1d_int(int *fv){
  free(fv);
}

float *array1d_float(long nc){

  float *fv;

  fv = (float*) malloc(nc * sizeof(float));
  if(!fv){
    fprintf(stderr,"Error: cannot allocate 1-D float vector\n");
    exit(1);
  }

  return fv;
}

void free_array1d_float(float *fv){
  free(fv);
}

fcomplex *array1d_fcomplex(long nc){

  fcomplex *fcv;

  fcv = (fcomplex*) malloc(nc * sizeof(fcomplex));
  if(!fcv){
    fprintf(stderr,"Error: cannot allocate 1-D float complex vector\n");
    exit(1);
  }

  return fcv;

}

void free_array1d_fcomplex(fcomplex *fcv){
  free(fcv);
}


float *vector_float(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
  float *v;

  v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
  if (!v){
    fprintf(stderr,"Error: cannot allocate 1-D vector\n");
    exit(1);  
  }
  
  return v-nl+NR_END;
}

void free_vector_float(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
  free((FREE_ARG) (v+nl-NR_END));
}



FILE *openfile(char *filename, char *pattern){
  FILE *fp;
  
  fp=fopen(filename, pattern);
  if (fp==NULL){
    fprintf(stderr,"Error: cannot open file: %s\n", filename);
    exit(1);
  }

  return fp;
}

void readdata(void *data, size_t blocksize, FILE *fp){
  if(fread(data, blocksize, 1, fp) != 1){
    fprintf(stderr,"Error: cannot read data\n");
    exit(1);
  }
}

void writedata(void *data, size_t blocksize, FILE *fp){
  if(fwrite(data, blocksize, 1, fp) != 1){
    fprintf(stderr,"Error: cannot write data\n");
    exit(1);
  }
}

long file_length(FILE* fp, long width, long element_size){
  long length;
  
  fseeko(fp,0L,SEEK_END);
  length = ftello(fp) / element_size / width;
  rewind(fp);
  
  return length;
}


// complex operations
fcomplex cmul(fcomplex a, fcomplex b)
{
  fcomplex c;
  c.re=a.re*b.re-a.im*b.im;
  c.im=a.im*b.re+a.re*b.im;
  return c;
}


long next_pow2(long a){
  long i;
  long x;
  
  x = 2;
  while(x < a){
    x *= 2;
  }
  
  return x;
}


void left_shift(fcomplex *in, int na){

  int i;
  fcomplex x;

  if(na < 1){
    fprintf(stderr, "Error: array size < 1\n\n");
    exit(1);
  }
  else if(na > 1){
    x.re = in[0].re;
    x.im = in[0].im;
    for(i = 0; i <= na - 2; i++){
      in[i].re = in[i+1].re;
      in[i].im = in[i+1].im;
    }
    in[na-1].re = x.re;
    in[na-1].im = x.im;  
  }
  else{ //na==1, no need to shift
    i = 0;
  }
}

void right_shift(fcomplex *in, int na){

  int i;
  fcomplex x;

  if(na < 1){
    fprintf(stderr, "Error: array size < 1\n\n");
    exit(1);
  }
  else if(na > 1){
    x.re = in[na-1].re;
    x.im = in[na-1].im;
    for(i = na - 1; i >= 1; i--){
      in[i].re = in[i-1].re;
      in[i].im = in[i-1].im;
    }
    in[0].re = x.re;
    in[0].im = x.im;
  }
  else{ //na==1, no need to shift
    i = 0;
  }
}


void circ_shift(fcomplex *in, int na, int nc){

  int i;
  int ncm;

  ncm = nc%na;

  if(ncm < 0){
    for(i = 0; i < abs(ncm); i++)
      left_shift(in, na);
  }
  else if(ncm > 0){
    for(i = 0; i < ncm; i++)
      right_shift(in, na);
  }
  else{ //ncm == 0, no need to shift
    i = 0;
  }
}


float bessi0(float x)
{
  float ax,ans;
  double y;

  if ((ax=fabs(x)) < 3.75) {
    y=x/3.75;
    y*=y;
    ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
      +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
  } else {
    y=3.75/ax;
    ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
      +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
      +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
      +y*0.392377e-2))))))));
  }
  return ans;
}




void kaiser2(float beta, int n, float *coef){

  int i;
  int hn;
  float a;

  hn = (n - 1) / 2;

  for(i = -hn; i<=hn; i++){
    a = 1.0 - 4.0 * i * i / (n - 1.0) / (n - 1.0);
    coef[i] = bessi0(beta * sqrt(a)) / bessi0(beta);
  }
}


void bandpass_filter(float bw, float bc, int n, int nfft, int ncshift, float beta, fcomplex *filter){

  int i;
  float *kw;
  int hn;
  fcomplex bwx, bcx;

  hn = (n-1)/2;

  if(n > nfft){
    fprintf(stderr, "Error: fft length too small!\n\n");
    exit(1);
  }
  if(abs(ncshift) > nfft){
    fprintf(stderr, "Error: fft length too small or shift too big!\n\n");
    exit(1);
  }

  //set all the elements to zero
  for(i = 0; i < nfft; i++){
    filter[i].re = 0.0;
    filter[i].im = 0.0;
  }

  //calculate kaiser window
  kw = vector_float(-hn, hn);
  kaiser2(beta, n, kw);

  //calculate filter
  for(i = -hn; i <= hn; i++){
    bcx.re = cos(bc * 2.0 * PI * i);
    bcx.im = sin(bc * 2.0 * PI * i);

    if(i == 0){
      bwx.re = 1.0;
      bwx.im = 0.0;
    }
    else{
      bwx.re = sin(bw * PI * i) / (bw * PI * i);
      bwx.im = 0.0;
    }
    
    filter[i+hn] = cmul(bcx, bwx);

    filter[i+hn].re = bw * kw[i] * filter[i+hn].re;
    filter[i+hn].im = bw * kw[i] * filter[i+hn].im;
  }

  //circularly shift filter, we shift the filter to left.
  ncshift = -abs(ncshift);
  circ_shift(filter, nfft, ncshift);

  free_vector_float(kw, -hn, hn);
}


#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr
void four1(float data[], unsigned long nn, int isign)
{
  unsigned long n,mmax,m,j,istep,i;
  double wtemp,wr,wpr,wpi,wi,theta;
  float tempr,tempi;

  n=nn << 1;
  j=1;
  for (i=1;i<n;i+=2) {
    if (j > i) {
      SWAP(data[j],data[i]);
      SWAP(data[j+1],data[i+1]);
    }
    m=nn;
    while (m >= 2 && j > m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  mmax=2;
  while (n > mmax) {
    istep=mmax << 1;
    theta=isign*(6.28318530717959/mmax);
    wtemp=sin(0.5*theta);
    wpr = -2.0*wtemp*wtemp;
    wpi=sin(theta);
    wr=1.0;
    wi=0.0;
    for (m=1;m<mmax;m+=2) {
      for (i=m;i<=n;i+=istep) {
        j=i+mmax;
        tempr=wr*data[j]-wi*data[j+1];
        tempi=wr*data[j+1]+wi*data[j];
        data[j]=data[i]-tempr;
        data[j+1]=data[i+1]-tempi;
        data[i] += tempr;
        data[i+1] += tempi;
      }
      wr=(wtemp=wr)*wpr-wi*wpi+wr;
      wi=wi*wpr+wtemp*wpi+wi;
    }
    mmax=istep;
  }
}
#undef SWAP








































//#include "resamp.h"

int main(int argc, char *argv[]){

  FILE *infp;   //slave image to be resampled
  FILE *outfp;  //resampled slave image

  fcomplex *filter;
  fcomplex *in;
  fcomplex *out;
  fcomplex *tmp;
  fcomplex *tmp2;
  int *zeroflag;

  int nrg; //file width
  int naz; //file length

  int nfft; //fft length
  int nfilter; //filter length
  int hnfilter;

  float bw;
  float bc;
  float beta; //kaiser window beta

  int zero_cf;
  float offset;

  float sc; //constant to scale the data read in to avoid large values
            //during fft and ifft
  float cf_pha;
  float t;
  fcomplex cf;

  int nblock_in;
  int nblock_out;
  int num_block;
  int i_block;
  int nblock_in_last;
  int nblock_out_last;

  int i, j;



/*****************************************************************************/
  nfilter = 65;
  nfft = 1024;
  beta = 1.0;
  zero_cf = 0;
  offset = 0.0;

  sc = 10000.0;
/*****************************************************************************/


  if(argc < 6){
    fprintf(stderr, "\nusage: %s inputf outputf nrg bw bc [nfilter] [beta] [zero_cf]\n", argv[0]);
    fprintf(stderr, "\nmandatory:\n");
    fprintf(stderr, "  inputf:  input file\n");
    fprintf(stderr, "  outputf: output file\n");
    fprintf(stderr, "  nrg:     file width\n");
    fprintf(stderr, "  bw:      filter bandwidth divided by sampling frequency [0, 1]\n");
    fprintf(stderr, "  bc:      filter center frequency divided by sampling frequency\n");
    fprintf(stderr, "\noptional:\n");
    fprintf(stderr, "  nfilter: number samples of the filter (odd). Default: 65\n");
    fprintf(stderr, "  nfft:    number of samples of the FFT. Default: 1024\n");
    fprintf(stderr, "  beta:    kaiser window beta. Default: 1.0\n");
    fprintf(stderr, "  zero_cf: if bc != 0.0, move center frequency to zero? 0: Yes (Default). 1: No.\n");
    fprintf(stderr, "  offset:  offset (in samples) of linear phase for moving center frequency. Default: 0.0\n\n");
    exit(1);
  }

  //open files
  infp  = openfile(argv[1], "rb");
  outfp = openfile(argv[2], "wb");

  nrg  = atoi(argv[3]);
  naz  = file_length(infp, nrg, sizeof(fcomplex));
  printf("file width: %d, file length: %d\n\n", nrg, naz);

  bw = atof(argv[4]);
  bc = atof(argv[5]);
  
  if(argc > 6)
    nfilter = atoi(argv[6]);
  if(argc > 7)
    nfft = atoi(argv[7]);

  if(argc > 8)
    beta = atof(argv[8]);
  if(argc > 9)
    zero_cf = atoi(argv[9]);
  if(argc > 10)
    offset = atof(argv[10]);

  if(nfilter < 3){
    fprintf(stderr, "filter length: %d too small!\n", nfilter);
    exit(1);
  }
  if(nfilter % 2 != 1){
    fprintf(stderr, "filter length must be odd!\n");
    exit(1);
  }

  hnfilter = (nfilter - 1) / 2;

  nblock_in = nfft - nfilter + 1;

  nblock_in += hnfilter;
  if (nblock_in <= 0){
    fprintf(stderr, "fft length too small compared with filter length!\n");
    exit(1);
  }

  nblock_out = nblock_in - 2 * hnfilter;

  num_block = (nrg - 2 * hnfilter) / nblock_out;
  if((nrg - num_block * nblock_out - 2 * hnfilter) != 0){
    num_block += 1;
  }
  if((nrg - 2 * hnfilter) <= 0){
    num_block = 1;
  }
  if(num_block == 1){
    nblock_out_last = 0;
    nblock_in_last = nrg;
  }
  else{
    nblock_out_last = nrg - (num_block - 1) * nblock_out - 2 * hnfilter;
    nblock_in_last = nblock_out_last + 2 * hnfilter;
  }

  filter = array1d_fcomplex(nfft);
  in     = array1d_fcomplex(nrg);
  out    = array1d_fcomplex(nrg);
  tmp    = array1d_fcomplex(nfft);
  tmp2   = array1d_fcomplex(nfft);
  zeroflag = array1d_int(nrg);


  bandpass_filter(bw, bc, nfilter, nfft, (nfilter-1)/2, beta, filter);

  //relationship of nr and matlab fft
  //nr fft           matlab fft
  //  1      <==>     ifft()*nfft
  // -1      <==>     fft()


  four1((float *)filter - 1, nfft, -1);


  for(i = 0; i < naz; i++){

    if((i + 1) % 1000 == 0 || (i + 1) == naz)
      fprintf(stderr,"processing line: %6d of %6d\r", i+1, naz);
    if((i + 1) == naz)
      fprintf(stderr,"\n\n");
  
    //read data
    readdata((fcomplex *)in, (size_t)nrg * sizeof(fcomplex), infp);
    for(j = 0; j < nrg; j++){
      if(in[j].re != 0.0 || in[j].im != 0.0){
        zeroflag[j] = 1;
      }
      else{
        zeroflag[j] = 0;
      }
      in[j].re *= 1.0 / sc;
      in[j].im *= 1.0 / sc;
    }

    //process
    for(i_block = 0; i_block < num_block; i_block++){
      //zero out
      for(j = 0; j < nfft; j++){
        tmp[j].re = 0.0;
        tmp[j].im = 0.0;
      }

      //get data
      if(num_block == 1){
        for(j = 0; j < nrg; j++){
          tmp[j] = in[j];
        }
      }
      else{
        if(i_block == num_block - 1){
          for(j = 0; j < nblock_in_last; j++){
            tmp[j] = in[j+nblock_out*i_block];
          }
        }
        else{
          for(j = 0; j < nblock_in; j++){
            tmp[j] = in[j+nblock_out*i_block];
          }
        }
      }


      four1((float *)tmp - 1, nfft, -1);

      for(j = 0; j < nfft; j++)
        tmp2[j] = cmul(filter[j], tmp[j]);

      four1((float *)tmp2 - 1, nfft, 1);


      if(num_block == 1){
        for(j = 0; j < nrg; j++){
          out[j] = tmp2[j];
        }
      }
      else{
        if(i_block == 0){
          for(j = 0; j < hnfilter + nblock_out; j++){
            out[j] = tmp2[j];
          }
        }
        else if(i_block == num_block - 1){
          for(j = 0; j < hnfilter + nblock_out_last; j++){
            out[nrg - 1 - j] = tmp2[nblock_in_last - 1 - j];
          }
        }
        else{
          for(j = 0; j < nblock_out; j++){
            out[j + hnfilter + i_block * nblock_out] = tmp2[j + hnfilter];
          }
        }
      }
    }

    //write data
    //move center frequency
    if(bc != 0 && zero_cf == 0){
      for(j = 0; j < nrg; j++){
        //t = j - (nrg - 1.0) / 2.0; //make 0 index exactly at range center
        t = j + offset; //make 0 index exactly at range center
        cf_pha = 2.0 * PI * (-bc) * t;
        cf.re = cos(cf_pha);
        cf.im = sin(cf_pha);
        out[j] = cmul(out[j], cf);
      }
    }
    //scale back
    for(j = 0; j < nrg; j++){
      if(zeroflag[j] == 0){
        out[j].re *= 0.0;
        out[j].im *= 0.0;
      }
      else{
        out[j].re *= sc / nfft;
        out[j].im *= sc / nfft;     
      }
    }
    //write data
    writedata((fcomplex *)out, nrg * sizeof(fcomplex), outfp);
  }

  free_array1d_fcomplex(filter);
  free_array1d_fcomplex(in);
  free_array1d_fcomplex(out);
  free_array1d_fcomplex(tmp);
  free_array1d_fcomplex(tmp2);
  free_array1d_int(zeroflag);
  fclose(infp);
  fclose(outfp);

  return 0;
}//end main()


