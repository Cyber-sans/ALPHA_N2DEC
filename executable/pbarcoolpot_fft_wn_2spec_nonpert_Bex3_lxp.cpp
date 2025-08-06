//this program calculates the self consistent potential for a plasma
//with 2 species (where both species can have similar numbers) in 
//a uniform B field and it assumes cylindrical symmetry
//the method is a simple relaxation: (1) calculate spatial dependent 
//   electric potential with a guessed density, (2) compute the Maxw-Bolt 
//   density from this potential, (3) add a small fraction of the new 
//   density to the old to get slightly better density.
//The potential is solved for in cylindrical coordinates by doing an 
//   expansion in sin(k z) F_k (r)
//uses 3 pt differencing in the rho direction
//uses sin fft in the z direction (hence should shift all voltages so 
//   ends are 0 and number of points in z should be 2^n); sinft() is from 
//   Numerical Recipes
//
// Important: does not get the electric field near the end of the range
//   accurately
//
// z/r dependent data are in outpotzfft_#.dat/outpotrfft_#.dat
//    where # is the number of the iteration so you can see how
//    the parameters are changing
// the edge of the plasma is output in contourfft_#.dat
//
// the total final data is output in efd_pot_tot.dat
//
//standard header files
#include <stdlib.h>
#include <string.h>  //string manipulations
#include <stdio.h>   // file I/O
#include <math.h>    // alternate math functions

//constants that are needed in the calculation
const double pi = double(2)*asin(double(1)) ;
const double twopi = double(2)*pi ;

//set the number of z points (must be 2^n) and number of radial points
const int numz = 8192*2 , numr = 512*4 ;

//note that the potential and densities are not computed for all points
//make sure that the full range of the density is covered
double poten[(numz/2)*(numr+1)], densit[(numz/2)*(numr+1)], dens[(numz/2)*(numr+1)],
       array2d[numz+1][numr+1], twodarr[numr+1][numz+1], densit2[(numz/2)*(numr+1)],
       densit3[(numz/2)*(numr+1)], dens2[(numz/2)*(numr+1)], densit4[(numz/2)*(numr+1)],
       densit5[(numz/2)*(numr+1)] ;

int main(int argc, char *argv[])
{
//the sin FFT function from numerical recipes
void sinft(double y[], int n) ;

//declare all of the variables used in this program
int idum, imid, iswitch, itemp, irun, nelem ;
long int i, j, k, iout, kzmin, kzmax, kmin, kmax, jfin,
         ist, ifi, ifir, kmaxr ;
double diag[numr+1], src[numr+1], offlu[numr+1], dum,
       edge[numz+1], eps0, dr, work[numr+1], dr2, test,
       rfin, diam, vt[30], lt[30], zfin, dz, kz, densmax,
       sum, z, prefz, offl[numr+1], offu[numr+1], echrg,
       kboltz, tempe, kt, rhofac, dens0, npart, nprot, bmag,
       zmid, sumden, norm, convfac, edgekz[numz+1], tempfac, energy, dum2,
       ztl, zth, rth, expfac, debye, omega, test0, test1, test2, densmax2,
       omegac, omegap, melec, consang, acc, rth2, mion, sumden2, norm2 ;
char fnm2[80], tmplt1[] = "outpotrfft_", suffx[] = ".dat",
     tmplt2[] = "outpotzfft_", tmplt6[] = "contourfft_",
     tmplt7[] = "efd_pot_tot_", tmplt8[] = "field_inp_2sp_",
     tmplt9[] = "2spc_z_int" ;
char *prog_name;

FILE *outcon ;

prog_name = argv[0];

if (argc<=3 && argc>2) {
  irun = atoi(argv[1]);
  itemp = atoi(argv[2]); }
else {
  fprintf(stdout,"Usage %s <run number> <temperature (int)>  \n",prog_name);
  exit(1);
}

//printf("\n\n run number, type in temperature (as integer) and hit return\n") ;
//scanf("%i %i",&irun,&itemp) ; getchar() ;
//accept command line argument:
//
if (argc != 3) {
    printf("Usage: %s <run_number> <temperature>\n", argv[0]);
    return 1;
}
irun = atoi(argv[1]);    // run number, e.g., 1001
itemp = atoi(argv[2]);   // temperature in Kelvin, e.g., 6
//

fprintf(stdout,"\n read in %i for the temperature\n\n",itemp) ;

sprintf(fnm2,"%s%i%s",tmplt8,irun,suffx) ;
printf("Reading from : %s\n",fnm2);
FILE *inpfiel ; inpfiel = fopen(fnm2,"r") ;

if(inpfiel == NULL) {fprintf(stdout,"\n\n HEY!!! Where is the input data file?\n\n KILL and start over\n") ; exit(1) ;}

//tempe is the plasma temperature in K
//all other parameters are named what they are
eps0 = 8.854e-12 ; echrg = 1.602E-19 ; bmag = 1.00 ; //1 Tesla B-field (Can change it to desired value)
tempe = double(itemp) ; kboltz = 1.381E-23 ; kt = tempe*kboltz ;
melec = 9.109e-31 ; mion = 9*1.673e-27 ;

//the following blocks hold the data for various machines and set ups
//choose the situation you want to investigate

//nelem is the number of electrodes
//vt holds the voltage on each electrode
//lt holds the length of each electrode (and is then converted to position)
//diam is the diameter of the electrode
//dens0 is the peak density of the positrons
//npart is the number of positrons
//nprot is the number of protons
//imid gives the number of the electrode before the position of the plasma
//zmid is the estimated z-position of the middle of the positron plasma

fscanf(inpfiel,"%i %i %lf %lf %lf %lf %lf %lf",&nelem,&imid,&dens0,&npart,&nprot,&tempfac,&acc,&diam) ;
imid -= 2 ;
for(j = 0 ; j < nelem ; j++)
{
fscanf(inpfiel,"%lf %lf",&dum,&dum2) ;
lt[j] = dum ; vt[j] = dum2 ;
}

//make sure the potentials are 0 at the edge of the range
dum = vt[0] ;
for(j = 0 ; j < nelem ; j++) vt[j] -= dum ;

//lt[j] is now the z-position of the final point of electrode j
for(j = 1 ; j < nelem ; j++) lt[j] += lt[j-1] ;

//zmid is the estimated middle position of the positron plasma
//IMPORTANT: this changes with configuration and machine
//           make sure the zmid is at the center of the positron density
ztl = lt[imid] ; zth = lt[imid+1] ; zmid = 0.5*(ztl+zth) ;

//rth is the radius of the trap
rth = diam/2 ;

//zfin is the final z of the trap, rfin is the radius of the trap
zfin = lt[nelem-1] ; rfin = diam/2 ;

//dr and dz are the step in r and z
dr = rfin/numr ; dr2 = dr*dr ; dz = zfin/numz ;

//check whether dz and dr are larger or smaller than the debye length
debye = sqrt(4.77e3*tempe/dens0) ;
fprintf(stdout,"dr = %13.6E dz = %13.6e  debye len = %13.6e\n",dr,dz,debye) ;
if((debye < dz) || (debye < dr))
{fprintf(stdout,"\n !!! possible instability problems because dr or dz > debye len !!!\n dont blame me if it doesnt converge\n hit return to continue\n") ; getchar() ;}

//kzmin and kzmax are the minimum and maximum range of the positron plasma
j = (long int)(zmid/dz) ;
kzmin = j + 2 - numz/4 ;
kzmax = j - 2 + numz/4 ;
if((kzmax - kzmin) > (numz/2))
{
fprintf(stdout,"problem!!  %i %i \n",int(kzmax -kzmin),int(numz)/2) ; exit(1) ;
}
fprintf(stdout,"kzmin = %i kzmax = %i \n",int(kzmin),int(kzmax)) ;

//jfin is the total number of points needed for the plasma
jfin = (kzmax-kzmin)*(numr+1) ;
//initialize the density to 0
for(j = 0 ; j <= jfin ; j++) {dens[j] = 0. ; dens2[j]=0. ;}

//kmin and kmax are the starting minimum and maximum range of the plasma
//kmin and kmax are adjusted as needed
kmin = 0.5*(kzmin+kzmax) - 400 ; kmax = kmin + 801 ;
//kmin = 1950 - 100 ; kmax = kmin + 201 ;
fprintf(stdout,"kmin = %i kmax = %i zmid = %15.8E \n",int(kmin),int(kmax),zmid) ;

//the cyclotron frequency, plasma frequency and 
//   rotational frequency of the plasma
omegac = echrg*bmag/melec ;
omegap = echrg*sqrt(dens0/(melec*eps0)) ;
omega = 0.5*(omegac-sqrt(omegac*omegac-omegap*omegap*2.0)) ;
fprintf(stdout,"omega_cyc = %13.6E omegap = %13.6E omega = %13.6E\n",omegac,omegap,omega) ;

//edge holds the z-dependent voltage at the electrodes
edge[1] = 0.0 ; i = 0 ;
for(j = 1 ; j <= numz; j++)
{
z = dz*j ; if(z > lt[i]) i++ ;
edgekz[j+1] = vt[i] ;
}

//edgekz holds the Fourier transform of the electrode voltage
idum = numz ;
sinft(edgekz,idum) ;
 prefz = dz*sqrt(2./zfin) ;
for(k = 1 ; k < numz ; k++)
{
kz = k*pi/zfin ; edgekz[k+1] *= prefz ;
}

//rhofac is the radial factor that goes into the Maxwell distribution to give
//    thermal equilibrium with a peak density of dens0 if only 1 species
rhofac = -dens0*echrg*echrg*0.25/eps0 ;
fprintf(stdout,"rhofac = %15.8E \n",rhofac) ;

//printf("\n finish set up, hit return to go\n") ; getchar() ;
fprintf(stdout,"\n !!!!!!!!!!!!!!!!!!!!!!\n   set up is finished\n !!!!!!!!!!!!!!!!!!!!!!\n \n") ;

//the off arrays hold the tridiagonal elements for solving the radial
//    potential
  offu[0] = 4.0/dr2 ;
  for(j = 1; j <= numr; j++)
  {
  offl[j-1] = (1.0-0.5/j)/dr2 ;
  offu[j] = (1.0+0.5/j)/dr2 ;
  }

  for(j = 0 ; j <= numr ; j++) offlu[j] = offl[j]*offu[j] ;
//
//initialize various parameters
iout = 0 ; test = 1.0 ; kmaxr = numr ;

//tempfac gives the temperature factor for the start of the run
//tempfac goes in as exp(-tempfac* e V/k_B T)
//          (smaller temperatures are easier to converge)
//tempfac increases with iterations until it is 1
expfac = 0.001 ; iswitch = 0 ;

//this is the iteration loop, keep iterating while "error" is > acc
//   or tempfac hasn't reached 1
while((test > acc) || (tempfac < 0.999))
{
//iout is the iteration number
iout ++ ;

//make sure enough points have been allocated for the arrays
if(kmin < kzmin) {fprintf(stdout,"stop!! kmin out of range\n") ; getchar() ; kmin = kzmin+1 ;}
if(kmax > kzmax) {fprintf(stdout,"stop!! kmax out of range\n") ; getchar() ; kmax = kzmax-1 ;}

if(kmin < kzmin) {kmin = kzmin+1 ;}
if(kmax > kzmax) {kmax = kzmax-1 ;}

//jfin is the total number of points needed in the arrays
jfin = (kzmax-kzmin)*(numr+1) ;

// initialize the potential
  for(j = 0 ; j <= jfin ; j++) poten[j] = 0. ;

//ist and ifi are the starting and finishing z-indexes
ist = kmin-kzmin ; ifi = kmax-kzmin ;
//Fourier transform the density
//twodarr will hold the source of the potential
for(j = 0 ; j <= numr ; j++)
{
  for(i = 0 ; i <= numz ; i++) edge[i] = 0.0 ;
 prefz = dz*sqrt(2./zfin)*echrg/eps0 ;
  for(i = ist ; i <= ifi ; i++) edge[i+kzmin+1] = -prefz*(dens[j+numr*i]+dens2[j+numr*i]) ;
 idum = numz ;
 sinft(edge,idum) ;
  for(i = 1 ; i <= numz ; i++) twodarr[j][i] = edge[i] ;
}

//this loop computes the Fourier transform of the potential in z
for(k = 1 ; k < numz ; k++)
{
kz = k*pi/zfin ;
for(j = 0 ; j <= numr ; j++) src[j] = twodarr[j][k+1] ;

  diag[0] = 1.0/(-(4.0/dr2) - kz*kz) ;
  dum = -(2.0/dr2) - kz*kz ;

  for(j = 1; j <= numr ; j++)
  {
  diag[j] = 1.0/(dum - offlu[j-1]*diag[j-1]) ;
  src[j] -= offl[j-1]*src[j-1]*diag[j-1] ;
  }

  twodarr[numr][k+1] = edgekz[k+1] ;
  for(j = numr ; j > 0 ; j--) twodarr[j-1][k+1] =
                       (src[j-1] - offu[j-1]*twodarr[j][k+1])*diag[j-1] ;
}

//this loop converts from k to z to get the potential as a function of
//      position
for(j = 0 ; j < kmaxr ; j++)
{
 prefz=sqrt(2./zfin) ;
  for(k = 1 ; k <= numz ; k++) edge[k] = prefz*twodarr[j][k] ;
 idum = numz ;
 sinft(edge,idum) ;

  for(i = ist ; i <= ifi ; i++) poten[j+numr*i] = edge[i+kzmin+1] ;
}

//adjust tempfac slowly to give stable convergence
if(test < 1.e-3) tempfac *= 1.01 ;
if(tempfac > 1.0) tempfac = 1.0 ;

if((iout%20) == 0)
fprintf(stdout,"tempfac = %8.3E convfac = %10.5E error = %8.3E expfac = %8.3E\n",
tempfac,convfac,test,expfac) ;

//for outputs, k is the estimated middle of the positron plasma in z
k = (kmin+kmax)/2 ;

//every 20 iterations output cuts along r and z
if((iout%20) == 0)
{
  //output z info
  sprintf(fnm2,"%s00%s",tmplt2,suffx) ; // last was %d int(iout)
  FILE *outgoz ; outgoz = fopen(fnm2,"w") ;
  for(j = kmin ; j < kmax ; j++)fprintf(outgoz,"%15.8E %15.8E %15.8E %15.8E %15.8E %15.8E\n",j*dz,
  poten[0+numr*(j-kzmin)],dens[0+numr*(j-kzmin)],densit[0+numr*(j-kzmin)],
  dens2[0+numr*(j-kzmin)],densit4[0+numr*(j-kzmin)]) ;
  fclose(outgoz) ;
  //finish z output

  sumden = 0.0 ;
   for(i = kmin ; i < kmax ; i++)
   {
    for(j = 0 ; j < kmaxr ; j++)
    {
    sumden += dr*(dr*j)*dz*dens[j+numr*(i-kzmin)]*twopi*exp((mion-melec)*omega*omega*j*j*dr*dr*0.5*tempfac/kt) ;
    }
   }
  consang = nprot/sumden ;
  //output r info
  sprintf(fnm2,"%s00%s",tmplt1,suffx) ;
  FILE *outgor ; outgor = fopen(fnm2,"w") ;
  for(j = 0 ; j < kmaxr ; j++)fprintf(outgor,"%15.8E %15.8E %15.8E %15.8E %15.8E %15.8E %15.8E %15.8E\n",
   j*dr,poten[j+numr*(k-kzmin)],dens[j+numr*(k-kzmin)],densit[j+numr*(k-kzmin)],dens2[j+numr*(k-kzmin)],densit4[j+numr*(k-kzmin)],
   (echrg*(poten[j+numr*(k-kzmin)]-poten[0+numr*(k-kzmin)])-rhofac*j*j*dr*dr)*tempfac/kt,
   consang*dens[j+numr*(k-kzmin)]*exp((mion-melec)*omega*omega*j*j*dr*dr*0.5*tempfac/kt)) ;
  fclose(outgor) ;
  //finish r output

  //output line integral of the positron and ion density
  sprintf(fnm2,"%s00%s",tmplt9,suffx) ;
  outgor = fopen(fnm2,"w") ;
  for(j = 0 ; j < kmaxr ; j++) {
    sumden = 0.0 ; densmax = 0.0 ;
    for(i = kmin ; i < kmax ; i++) {
       // sumden += consang*dz*dens[j+numr*(i-kzmin)]*twopi*exp((mion-melec)*omega*omega*j*j*dr*dr*0.5*tempfac/kt) ;
      sumden += dz*dens2[j+numr*(i-kzmin)]*twopi ;
      densmax += dz*dens[j+numr*(i-kzmin)]*twopi ;
    }
    fprintf(outgor,"%13.6E %13.6E %13.6E\n",(j*dr),densmax,sumden) ; 
  }
  fclose(outgor) ;
  //finish output of line integral of positron and proton density


  //output contour info
  sprintf(fnm2,"%s00%s",tmplt6,suffx) ; outcon = fopen(fnm2,"w") ;
  //first find the maximum density
  densmax = 0.0 ; sumden = 0.0 ; consang = 0.0 ;
  densmax2 = 0.0 ;
  for(i = kmin ; i <= kmax ; i++) {
    for(j = 0 ; j < kmaxr ; j++) {
      if(dens[j+numr*(i-kzmin)] > densmax) densmax = dens[j+numr*(i-kzmin)] ;
      if(dens2[j+numr*(i-kzmin)] > densmax2) densmax2 = dens2[j+numr*(i-kzmin)] ;
      sumden += dr*(dr*j)*dz*dens[j+numr*(i-kzmin)]*twopi ;
      consang +=dr*(dr*j)*dz*dens[j+numr*(i-kzmin)]*twopi*(dr*j)*(dr*j) ;
    }
  }
  consang /= sumden ;

  //next find position where density is 1/2 the max density at r = 0
  //ztl is the lower z where density is 1/2 max and zth is the higher z
  densmax *= 0.5 ; densmax2 *= 0.5 ;
  j = 0 ; dum = -densmax ;
  for(i = kmin+1 ; i <= kmax ; i++) {
    dum2 = dum ; dum = dens[j+numr*(i-kzmin)] - densmax;
    if((dum2 < 0.0) && (dum > 0.0)) ztl = i*dz - dz*dum/(dum-dum2) ;
    if((dum2 > 0.0) && (dum < 0.0)) zth = i*dz - dz*dum/(dum-dum2) ;
  }
  dum = 0.5 + 0.5*(ztl+zth)/dz ; i = int(dum) ;

  //now find how wide in r the plasma is at the middle of the plasma
  j = 0 ; dum = densmax ;
  for(j = 1 ; j < kmaxr ; j++) {
    dum2 = dum ; dum = dens[j+numr*(i-kzmin)] - densmax;
    if((dum2 > 0.0) && (dum < 0.0)) rth = j*dr - dr*dum/(dum-dum2) ;
  }

  //output to the screen plasma size parameters
  //middle position of plasma, r_max of plasma, z_width of plasma,
  //estimated density using spheroid, v at edge
  //   printf("z = %10.3E r =%10.3E Dz =%10.3E d_est =%12.5E v =%10.3E\n",0.5*(ztl+zth),
  //   rth,(zth-ztl),3*npart/(2.0*pi*rth*rth*(zth-ztl)),omega*rth) ;
  printf("z = %10.3E r =%10.3E Dz =%10.3E d_est =%12.5E L =%10.3E\n",0.5*(ztl+zth),
  rth,(zth-ztl),3*npart/(2.0*pi*rth*rth*(zth-ztl)),consang) ;
   
/*
//now output the positions where the density is 1/2, this lets you draw 
//   outer edge of plasma
  for(i = kmin ; i <= kmax ; i++)
  {
    for(j = 1 ; j < kmaxr ; j++)
    {
    if((dens[j-1+numr*(i-kzmin)] > densmax) &&
    (dens[j+numr*(i-kzmin)] < densmax))
    { fprintf(outcon,"%15.8E %15.8E\n",i*dz,j*dr) ;
      fprintf(outcon,"%15.8E %15.8E\n",i*dz,-j*dr) ;}
    }
  }
*/

  //now output the two densities at every other point
  for(i = kmin ; i <= kmax ; i+=2)
  {
    for(j = -kmaxr ; j < kmaxr ; j+=2)
    {
    if(j < 0) fprintf(outcon,"%11.4E %11.4E %11.4E %11.4E\n",i*dz-zmid,j*dr,dens[-j+numr*(i-kzmin)],dens2[-j+numr*(i-kzmin)]) ;
    else fprintf(outcon,"%11.4E %11.4E %11.4E %11.4E\n",i*dz-zmid,j*dr,dens[j+numr*(i-kzmin)],dens2[j+numr*(i-kzmin)]) ;
    }
  //next line needed for FJR plot in gnuplot, puts in a line skip for every z
  fprintf(outcon,"\n") ;
  }
  fclose(outcon) ;
  //finish output of contour info
}
//sfinish block that outputs info every 10 iterations

//use the current potential to give the Maxwell density (densit & densit4)
//  densit holds the positron density and densit4 holds the ion density
//Compute normalization of densit & densit4
sumden = 0. ; k = (kzmin+kzmax)/2 ; energy = 0.0 ; densmax = 0.0 ; rth2 = 0.0 ;
sumden2 = 0. ;
  for(i = kmin ; i <= kmax ; i++)
  {
    for(j = 0 ; j < kmaxr ; j++)
    {
    dum = (echrg*(poten[j+numr*(i-kzmin)]-poten[0+numr*(k-kzmin)])-rhofac*j*j*dr*dr)*tempfac/kt ;
    densit[j+numr*(i-kzmin)] = 0.0 ;
    rth2 += dr*dr*j*twopi*dz*dens[j+numr*(i-kzmin)]*dr*j*dr*j ;
    energy += 0.5*dr*dr*j*twopi*dz*dens[j+numr*(i-kzmin)]*poten[j+numr*(i-kzmin)] ;
      if(dum < 36.) densit[j+numr*(i-kzmin)] = exp(-dum) ;
    dum  -= (mion-melec)*omega*omega*j*j*dr*dr*0.5*tempfac/kt ;
    densit4[j+numr*(i-kzmin)] = 0.0 ;
      if(dum < 36.) densit4[j+numr*(i-kzmin)] = exp(-dum) ;
//      if((dum < 36.) && (i*dz > 0.96)) densit[j+numr*(i-kzmin)] = exp(-dum) ;
      if(densit[j+numr*(i-kzmin)] > densmax) densmax = densit[j+numr*(i-kzmin)] ;
    sumden += dr*(dr*j)*dz*densit[j+numr*(i-kzmin)]*twopi ;
    sumden2 += dr*(dr*j)*dz*densit4[j+numr*(i-kzmin)]*twopi ;
    }
  }
norm = npart/sumden ; norm2 = nprot/sumden2 ;
if(iout == 1) expfac = 0.5*dens0/(densmax*norm) ;

//now starts the block where I estimate by how much to change the density
//so that I get the best convergence
//the change is made by adding some of densit to dens
//expfac is the expected fraction that will need to be added

convfac = 1.0-expfac ;

//test0 is the error from using 0 for the change of the density
//densit2 is the density changed by 2*expfac
sum = 0.0 ; dum = 0.0 ;
  for(i = kmin ; i <= kmax ; i++)
  {
   for(j = 0 ; j < kmaxr ; j++)
   {
   densit[j+numr*(i-kzmin)] *= norm ;
   densit4[j+numr*(i-kzmin)] *= norm2 ;
//   densit2[j+numr*(i-kzmin)] = convfac*dens[j+numr*(i-kzmin)]+expfac*densit[j+numr*(i-kzmin)] ;
   densit2[j+numr*(i-kzmin)] = (1-2*expfac)*(dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)])+
   2*expfac*(densit[j+numr*(i-kzmin)]+densit4[j+numr*(i-kzmin)]) ;
   sum += (dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)]-densit[j+numr*(i-kzmin)]-densit4[j+numr*(i-kzmin)])*
    (dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)]-densit[j+numr*(i-kzmin)]-densit4[j+numr*(i-kzmin)])*j ;
   dum += ((dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)])*(dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)])+
    (densit[j+numr*(i-kzmin)]+densit4[j+numr*(i-kzmin)])*(densit[j+numr*(i-kzmin)]+densit4[j+numr*(i-kzmin)]))*j ;
    }
  }
//test0 is the parameter used to track the error in the density
//  and is the integral of the squared difference in density divided
//  by the integral of the squared density
  test0 = sum/dum ;
    

//compute the potential from the density densit2

//jfin is the total number of points needed in the arrays
jfin = (kzmax-kzmin)*(numr+1) ;

// initialize the potential
  for(j = 0 ; j <= jfin ; j++) poten[j] = 0. ;

//ist and ifi are the starting and finishing z-indexes
ist = kmin-kzmin ; ifi = kmax-kzmin ;
//Fourier transform the density
//twodarr will hold the source of the potential
for(j = 0 ; j <= numr ; j++)
{
  for(i = 0 ; i <= numz ; i++) edge[i] = 0.0 ;
 prefz = dz*sqrt(2./zfin)*echrg/eps0 ;
  for(i = ist ; i <= ifi ; i++) edge[i+kzmin+1] = -prefz*densit2[j+numr*i] ;
 idum = numz ;
 sinft(edge,idum) ;
  for(i = 1 ; i <= numz ; i++) twodarr[j][i] = edge[i] ;
}

//this loop computes the Fourier transform of the potential in z
for(k = 1 ; k < numz ; k++)
{
kz = k*pi/zfin ;
for(j = 0 ; j <= numr ; j++) src[j] = twodarr[j][k+1] ;

  diag[0] = 1.0/(-(4.0/dr2) - kz*kz) ;
  dum = -(2.0/dr2) - kz*kz ;

  for(j = 1; j <= numr ; j++)
  {
  diag[j] = 1.0/(dum - offlu[j-1]*diag[j-1]) ;
  src[j] -= offl[j-1]*src[j-1]*diag[j-1] ;
  }

  twodarr[numr][k+1] = edgekz[k+1] ;
  for(j = numr ; j > 0 ; j--) twodarr[j-1][k+1] =
                       (src[j-1] - offu[j-1]*twodarr[j][k+1])*diag[j-1] ;
}

//this loop converts from k to z to get the potential as a function of
//      position
for(j = 0 ; j < kmaxr ; j++)
{
 prefz=sqrt(2./zfin) ;
  for(k = 1 ; k <= numz ; k++) edge[k] = prefz*twodarr[j][k] ;
 idum = numz ;
 sinft(edge,idum) ;

  for(i = ist ; i <= ifi ; i++) poten[j+numr*i] = edge[i+kzmin+1] ;
}

sumden = 0. ; k = (kzmin+kzmax)/2 ; densmax = 0.0 ; sumden2 = 0. ;
  for(i = kmin ; i <= kmax ; i++)
  {
    for(j = 0 ; j < kmaxr ; j++)
    {
    dum = (echrg*(poten[j+numr*(i-kzmin)]-poten[0+numr*(k-kzmin)])-rhofac*j*j*dr*dr)*tempfac/kt ;
    densit3[j+numr*(i-kzmin)] = 0.0 ;
      if(dum < 36.) densit3[j+numr*(i-kzmin)] = exp(-dum) ;
    dum -= (mion-melec)*omega*omega*j*j*dr*dr*0.5*tempfac/kt ;
    densit5[j+numr*(i-kzmin)] = 0.0 ;
      if(dum < 36.) densit5[j+numr*(i-kzmin)] = exp(-dum) ;
//      if((dum < 36.) && (i*dz > 0.96)) densit3[j+numr*(i-kzmin)] = exp(-dum) ;
    sumden += dr*(dr*j)*dz*densit3[j+numr*(i-kzmin)]*twopi ;
    sumden2 += dr*(dr*j)*dz*densit5[j+numr*(i-kzmin)]*twopi ;
    }
  }
norm = npart/sumden ; norm2 = nprot/sumden2 ;
convfac = 1.0-2.0*expfac ;
//finish the calculation of the potential using densit2

//compute the Maxwell-Boltzmann density using the potential from densit2
//test2 is the error error from using the change by 2*exp
sum = 0.0 ; dum = 0.0 ;
  for(i = kmin ; i <= kmax ; i++)
  {
   for(j = 0 ; j < kmaxr ; j++)
   {
   densit3[j+numr*(i-kzmin)] *= norm ;
   densit5[j+numr*(i-kzmin)] *= norm2 ;
   sum += (densit2[j+numr*(i-kzmin)]-densit3[j+numr*(i-kzmin)]-densit5[j+numr*(i-kzmin)])*
    (densit2[j+numr*(i-kzmin)]-densit3[j+numr*(i-kzmin)]-densit5[j+numr*(i-kzmin)])*j ;
   dum += (densit2[j+numr*(i-kzmin)]*densit2[j+numr*(i-kzmin)]+
    (densit3[j+numr*(i-kzmin)]+densit5[j+numr*(i-kzmin)])*(densit3[j+numr*(i-kzmin)]+densit5[j+numr*(i-kzmin)]))*j ;
    }
  }
  test2 = sum/dum ;
 
//update how large expfac should be
dum = 1.0 ;
if(test2 < test0) dum = 1.03 ;
if(test2 > test0) dum = 0.97 ;

expfac *= dum ;
convfac = 1.0-expfac ;

//update the density by an amount expfac
sumden = 0. ; densmax = 0. ; sum = 0.0 ; dum = 0.0 ; sumden2 = 0. ;
consang = 0. ; dum2 = 0. ;
  for(i = kmin ; i <= kmax ; i++)
  {
    for(j = 0 ; j < kmaxr ; j++)
    {
//decrease the current density by amount convfac
    dens[j+numr*(i-kzmin)] *= convfac ;
    dens2[j+numr*(i-kzmin)] *= convfac ;
//add to density the iterated density by amount (1-convfac)
    dens[j+numr*(i-kzmin)] += (1.0-convfac)*densit[j+numr*(i-kzmin)] ;
    dens2[j+numr*(i-kzmin)] += (1.0-convfac)*densit4[j+numr*(i-kzmin)] ;
//sum and dum are used to compute the "error"
   sum += (dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)]-densit[j+numr*(i-kzmin)]-densit4[j+numr*(i-kzmin)])*
    (dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)]-densit[j+numr*(i-kzmin)]-densit4[j+numr*(i-kzmin)])*j ;
   dum += ((dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)])*(dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)])+
    (densit[j+numr*(i-kzmin)]+densit4[j+numr*(i-kzmin)])*(densit[j+numr*(i-kzmin)]+densit4[j+numr*(i-kzmin)]))*j ;
//compute densmax which will be used to estimate range needed in calculation
    if((dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)]) > densmax) densmax = dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)] ;
//sumden is the total number of positrons in the plasma
//sumden2 is the total number of ions in the plasma
    sumden += dens[j+numr*(i-kzmin)]*twopi*dr*dr*j*dz ;
    sumden2 += dens2[j+numr*(i-kzmin)]*twopi*dr*dr*j*dz ;
//once the integral is completed consang/dum2 is a measure of overlap of the plasmas
    consang += 4*dens[j+numr*(i-kzmin)]*dens2[j+numr*(i-kzmin)]*twopi*dr*dr*j*dz ;
    dum2 += (dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)])*(dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)])*twopi*dr*dr*j*dz ;
    }
  }
//test is the "error" in the density which is used to control whether to continue iterating
  test = sum/dum ; ifir = 0 ;

//estimate the range of z and r needed by only keeping points where
//     dens > 1e-11*densmax
  densmax *= 1.0e-10*test ; ist = kmin ; ifi = kmax ; 
  kmin = kzmax ; kmax = kzmin ;
  
  for(i = ist ; i <= ifi ; i++)
  {
    for(j = 0 ; j < kmaxr ; j++)
    {
      if((dens[j+numr*(i-kzmin)]+dens2[j+numr*(i-kzmin)]) > densmax)
      {
      if(i < kmin) kmin = i ;
      if(i > kmax) kmax = i ;
      if(j > ifir) ifir = j ;
      }
    }
  }
//shift kmin and kmax by a little bit to allow size of range to increase
kmin -= 2 ; kmax += 2 ; kmaxr = ifir+2 ;
//if(kmin < 1790) kmin = 1790 ;
if(kmaxr > numr) kmaxr = numr ;

//output to the screen parameters
//iteration number, error in number of particles, kmin_z, kmax_z, kmax_r,
//    and electrostatic potential energy
if((iout%20) == 0)
printf("it=%i Dn =%9.2E %9.2E ovlp =%9.2E kmin =%i kmax =%i kmax_r =%i\n\n",
     int(iout),npart-sumden,nprot-sumden2,consang/dum2,int(kmin),
     int(kmax),int(kmaxr)) ;
//printf("it=%i Dn =%9.2E kmin =%i kmax =%i kmax_r =%i E =%10.3E  %10.3E\n\n",
//     int(iout),npart-sumden,int(kmin),
//     int(kmax),int(kmaxr),energy*echrg/(npart*kboltz),sqrt(3.0*rth2/npart)) ;
     
}//this is the end of the iteration loop


sprintf(fnm2,"%s00%s",tmplt6,suffx) ; outcon = fopen(fnm2,"w") ;
//now output the two densities at every other point
  for(i = kmin ; i <= kmax ; i++)
  {
    for(j = -kmaxr ; j < kmaxr ; j++)
    {
    if(j < 0) fprintf(outcon,"%11.4E %11.4E %11.4E %11.4E\n",i*dz-zmid,j*dr,dens[-j+numr*(i-kzmin)],dens2[-j+numr*(i-kzmin)]) ;
    else fprintf(outcon,"%11.4E %11.4E %11.4E %11.4E\n",i*dz-zmid,j*dr,dens[j+numr*(i-kzmin)],dens2[j+numr*(i-kzmin)]) ;
    }
//next line needed for FJR plot in gnuplot, puts in a line skip for every z
  fprintf(outcon,"\n") ;
  }
fclose(outcon) ;
//finish output of contour info


// New Change, by DY
// === Final output of line integral of the positron and ion density ===
sprintf(fnm2,"%s00%s",tmplt9,suffx) ;
FILE *outgor = fopen(fnm2,"w") ;
for(j = 0 ; j < kmaxr ; j++) {
    sumden = 0.0 ; densmax = 0.0 ;
    for(i = kmin ; i < kmax ; i++) {
        // sumden += consang*dz*dens[j+numr*(i-kzmin)]*twopi*exp((mion-melec)*omega*omega*j*j*dr*dr*0.5*tempfac/kt) ;
        sumden += dz*dens2[j+numr*(i-kzmin)]*twopi ;
        densmax += dz*dens[j+numr*(i-kzmin)]*twopi ;
    }
    fprintf(outgor,"%13.6E %13.6E %13.6E\n",(j*dr),densmax,sumden) ; 
}
fclose(outgor) ;
// === End of final output block ===


//the following block will outputall the data throughout the trap
//  this data is useful but slow so I have it commented out at this point
/*
//output for recombination code
//uses the calculated density to compute the potential everywhere in trap
sprintf(fnm2,"%s%i_%i%s",tmplt7,itemp,irun,suffx) ;
FILE *outtot ; outtot = fopen(fnm2,"w") ;
fprintf(outtot,"%i \n",numz) ;
fprintf(outtot,"%i \n",numr) ;
fprintf(outtot,"%13.6E \n",dz) ;
fprintf(outtot,"%13.6E \n",dr) ;
fprintf(outtot,"%13.6E \n",ztl) ;
fprintf(outtot,"%13.6E \n",zth) ;
fprintf(outtot,"%13.6E \n",rth) ;

//compute the potential
for(i = 0 ; i <= numz ; i++)
{
  for(j = 0 ; j < numr ; j++) array2d[i][j] = 0.0 ;
}
for(k = 1 ; k <= numz ; k++)
{
sum = 0. ; kz = k*pi/zfin ; prefz = dz*sqrt(2./zfin) ;
if(k == 0) prefz = dz/sqrt(zfin) ;

for(j = 0 ; j <= numr ; j++) src[j] = 0. ;

ist = kmin-kzmin ; ifi = kmax-kzmin ;
for(i = ist ; i <= ifi ; i++)
{
dum = prefz*sin((i+kzmin)*dz*kz)*echrg/eps0 ;
for(j = 0 ; j < numr ; j++) src[j] -= dum*(dens[j+numr*i]+dens2[j+numr*i]) ;
}

  diag[0] = 1.0/(-(4.0/dr2) - kz*kz) ;
  dum = -(2.0/dr2) - kz*kz ;

  for(j = 1; j <= numr ; j++)
  {
  diag[j] = 1.0/(dum - offlu[j-1]*diag[j-1]) ;
  src[j] -= offl[j-1]*src[j-1]*diag[j-1] ;
  }

  work[numr] = edgekz[k+1] ;
  for(j = numr ; j > 0 ; j--) work[j-1] =
                       (src[j-1] - offu[j-1]*work[j])*diag[j-1] ;

  prefz=sqrt(2./zfin) ; if(k == 0) prefz = 1./sqrt(zfin) ;

  for(i = 0 ; i <= numz ; i++)
  {
  dum = prefz*sin(kz*i*dz) ;
    for(j = 0 ; j < numr ; j++) array2d[i][j] += dum*work[j] ;
  }
}
//print out the potential
for(i = 0 ; i <= numz ; i++)
{
  for(j = 0 ; j < numr ; j++) fprintf(outtot,"%14.7E\n",array2d[i][j]) ;
}

//fill array2d with 0's then fill in nonzero positron density elements
for(i = 0 ; i <= numz ; i++)
{
  for(j = 0 ; j < numr ; j++) array2d[i][j] = 0.0 ;
}
ist = kmin-kzmin ; ifi = kmax-kzmin ;
for(i = ist ; i <= ifi ; i++)
{
for(j = 0 ; j < numr ; j++) array2d[i+kzmin][j] = dens[j+numr*i] ;
}
//print out the density
for(i = 0 ; i <= numz ; i++)
{
  for(j = 0 ; j < numr ; j++) fprintf(outtot,"%14.7E\n",array2d[i][j]) ;
}

//fill array2d with 0's then fill in nonzero ion density elements
for(i = 0 ; i <= numz ; i++)
{
  for(j = 0 ; j < numr ; j++) array2d[i][j] = 0.0 ;
}
ist = kmin-kzmin ; ifi = kmax-kzmin ;
for(i = ist ; i <= ifi ; i++)
{
for(j = 0 ; j < numr ; j++) array2d[i+kzmin][j] = dens2[j+numr*i] ;
}
//print out the density
for(i = 0 ; i <= numz ; i++)
{
  for(j = 0 ; j < numr ; j++) fprintf(outtot,"%14.7E\n",array2d[i][j]) ;
}

//compute the potential with no charges
for(i = 0 ; i <= numz ; i++)
{
  for(j = 0 ; j < numr ; j++) array2d[i][j] = 0.0 ;
}
for(k = 1 ; k <= numz ; k++)
{
sum = 0. ; kz = k*pi/zfin ; prefz = dz*sqrt(2./zfin) ;
if(k == 0) prefz = dz/sqrt(zfin) ;

for(j = 0 ; j <= numr ; j++) src[j] = 0. ;

ist = kmin-kzmin ; ifi = kmax-kzmin ;

//for(i = ist ; i <= ifi ; i++)
//{
//dum = prefz*sin((i+kzmin)*dz*kz)*echrg/eps0 ;
//for(j = 0 ; j < numr ; j++) src[j] -= dum*dens[j+numr*i] ;
//}


  diag[0] = 1.0/(-(4.0/dr2) - kz*kz) ;
  dum = -(2.0/dr2) - kz*kz ;

  for(j = 1; j <= numr ; j++)
  {
  diag[j] = 1.0/(dum - offlu[j-1]*diag[j-1]) ;
  src[j] -= offl[j-1]*src[j-1]*diag[j-1] ;
  }

  work[numr] = edgekz[k+1] ;
  for(j = numr ; j > 0 ; j--) work[j-1] =
                       (src[j-1] - offu[j-1]*work[j])*diag[j-1] ;

  prefz=sqrt(2./zfin) ; if(k == 0) prefz = 1./sqrt(zfin) ;

  for(i = 0 ; i <= numz ; i++)
  {
  dum = prefz*sin(kz*i*dz) ;
    for(j = 0 ; j < numr ; j++) array2d[i][j] += dum*work[j] ;
  }
}
//print out the potential with no charges
for(i = 0 ; i <= numz ; i++)
{
  for(j = 0 ; j < numr ; j++) fprintf(outtot,"%14.7E\n",array2d[i][j]) ;
}
*/

}

//Slightly modified Numerical Recipes functions for computing sin fft
void sinft(double y[], int n)
{
	void realft(double data[], unsigned long n, int isign);
	int j,n2=n+2;
	double sum,y1,y2;
	double theta,wi=0.0,wr=1.0,wpi,wpr,wtemp;

//	theta=3.14159265358979/(double) n;
	theta=pi/(double) n;
	wtemp=sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi=sin(theta);
	y[1]=0.0;
	for (j=2;j<=(n>>1)+1;j++) {
		wr=(wtemp=wr)*wpr-wi*wpi+wr;
		wi=wi*wpr+wtemp*wpi+wi;
		y1=wi*(y[j]+y[n2-j]);
		y2=0.5*(y[j]-y[n2-j]);
		y[j]=y1+y2;
		y[n2-j]=y1-y2;
	}
	realft(y,n,1);
	y[1]*=0.5;
	sum=y[2]=0.0;
	for (j=1;j<=n-1;j+=2) {
		sum += y[j];
		y[j]=y[j+1];
		y[j+1]=sum;
	}
}

void realft(double data[], unsigned long n, int isign)
{
	void four1(double data[], unsigned long nn, int isign);
	unsigned long i,i1,i2,i3,i4,np3;
	double c1=0.5,c2,h1r,h1i,h2r,h2i;
	double wr,wi,wpr,wpi,wtemp,theta;

//	theta=3.141592653589793/(double) (n>>1);
	theta=pi/(double) (n>>1);
	if (isign == 1) {
		c2 = -0.5;
		four1(data,n>>1,1);
	} else {
		c2=0.5;
		theta = -theta;
	}
	wtemp=sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi=sin(theta);
	wr=1.0+wpr;
	wi=wpi;
	np3=n+3;
	for (i=2;i<=(n>>2);i++) {
		i4=1+(i3=np3-(i2=1+(i1=i+i-1)));
		h1r=c1*(data[i1]+data[i3]);
		h1i=c1*(data[i2]-data[i4]);
		h2r = -c2*(data[i2]+data[i4]);
		h2i=c2*(data[i1]-data[i3]);
		data[i1]=h1r+wr*h2r-wi*h2i;
		data[i2]=h1i+wr*h2i+wi*h2r;
		data[i3]=h1r-wr*h2r+wi*h2i;
		data[i4] = -h1i+wr*h2i+wi*h2r;
		wr=(wtemp=wr)*wpr-wi*wpi+wr;
		wi=wi*wpr+wtemp*wpi+wi;
	}
	if (isign == 1) {
		data[1] = (h1r=data[1])+data[2];
		data[2] = h1r-data[2];
	} else {
		data[1]=c1*((h1r=data[1])+data[2]);
		data[2]=c1*(h1r-data[2]);
		four1(data,n>>1,-1);
	}
}

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

void four1(double data[], unsigned long nn, int isign)
{
	unsigned long n,mmax,m,j,istep,i;
	double wtemp,wr,wpr,wpi,wi,theta;
	double tempr,tempi;

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
		theta=isign*(twopi/mmax);
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
