#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

/*
** Dvect is a class set up to store and manipulate a double precision vector.
*/
class Dvect {
public:
  Dvect(void);                     // default constructor
  Dvect(int);                      // alternate constructor
  Dvect(const Dvect&);             // copy constructor
  ~Dvect();                        // destructor

  double element(int) const;       // gets the value of a specific element in the vector
  void   setall(double);           // sets all elements of a vector to the argument value
  void   set(int,double);          // sets a specific element to the argument value
  int    size(void) const;         // gets the size of the vector

  Dvect& operator*=(const double); // multiplies the vector by a constant
  Dvect  operator*(const double);  // multiplies a vector by a constant
  Dvect& operator*=(const Dvect&); // multiplies the vector element-by-element with another
  Dvect  operator*(const Dvect&);  // multiplies two vectors element-by-element
  Dvect& operator+=(const Dvect&); // adds the vector element-by-element with another
  Dvect  operator+(const Dvect&);  // adds two vectors together element-by-element
  Dvect& operator-=(const Dvect&); // subtracts another vector element-by-element from this one  
  Dvect  operator-(const Dvect&);  // subtracts two vectors element-by-element
  Dvect& operator=(const double);  // sets all elements of a vector to a specific value
  Dvect& operator=(const Dvect&);  // sets the elements to the same as those of another
  double operator[](int) const;    // allows accessing an individual element via brackets
  double& operator[](int);         // allows setting an individual element via brackets

  bool   resize(int);              // discards the data and sets the vector size to a new value
  bool   copy(const Dvect&);       // copies the data from an input vector to this one
  double sum(void);                // returns the summation of all elements of this vector

  friend ostream& operator<<(ostream&,const Dvect&); // outputs all elements to a stream
  friend istream& operator>>(istream&, Dvect&);      // inputs n elements from a stream

private:
  double *a;                       // the array for the vector data
  int    sz;                       // the size of the vector
};

// read_input reads the input files and stores the data in the Dvect class
bool   read_input(char **, Dvect&, Dvect&, Dvect**, bool=true);
// calculates the gradient from a weight, y-observations, and features
void   grad(Dvect&, Dvect&, Dvect&, Dvect*);
void   gradest(Dvect&, Dvect&, Dvect&, Dvect*);
// calculates the objective function
void   LLcomp(Dvect&, Dvect&, Dvect&, Dvect*);
double LL(Dvect&, Dvect&, Dvect*);
// calculate the next set of weights
void   getnext(Dvect&, Dvect&, double);
// iterate to a solution: 
// getsoln(weights, y-objserved, features, delta-criteria, max-iterations)
int    getsoln(Dvect&, Dvect&, Dvect*, double=0.001, int=100);

int main(int argv, char **argc) {

  cout << "Executing logr with command line arguments: ";
  for (int i=0; i<argv; i++)
	cout << argc[i] << " ";
  cout << endl;

  Dvect    wvec;          // weights input file
  Dvect    yvec;          // observed y-values
  Dvect   *xvec;          // features (array size is number of data points supported)
  Dvect    lvec;          // log liklihood function components
  int      niter;         // number of iterations used

  if (argv == 4) {
	if (read_input(argc, wvec, yvec, &xvec)) {
	  niter = getsoln(wvec, yvec, xvec, 0.01, 1000);
	  cout << "Solution after " << niter << " iterations: " << endl;
	  cout << wvec << endl;
	  cout << "Log Liklihood:" << endl;
	  LLcomp(lvec, wvec, yvec, xvec);
	  cout << lvec << " = " << lvec.sum() << endl;
	}
  }
  else {
	cout << "Usage:  ./logr [Initial Weights] [y-data] [x-data]" << endl;
  }

  return 0;
}

/*
** The getsoln() function iterates to a solution until either the objective
** function change from one iteration to the next is less than espsilon, or the
** max number of iterations is reached.
*/
int getsoln(Dvect &w, Dvect &y, Dvect *x, double epsilon, int maxiter) {
  int    i;            // counter
  double ll, ll_old;   // objective function values
  double alpha = 0.1;  // speed at which to converge using the gradient
  Dvect  dk(w.size()); // temp variable for the gradient

  ll = ll_old = 0.0;
  for (i=0; i<maxiter; i++) {
	//cout << endl;
	//cout << "Iteration #" << i << endl;
	//cout << "============" << endl;
	grad(dk, w, y, x);
	//cout << "  Gradient: " << dk << endl;
	//gradest(dk, w, y, x);
	//cout << "  Gradient Est: " << dk << endl;
	ll_old = ll;
	ll     = LL(w, y, x);
    if (fabs(ll_old-ll) < epsilon) break;
	//cout << "  Objective function: " << ll << endl;
	//cout << "  diff:               " << fabs(ll_old-ll) << endl;
	getnext(w, dk, alpha);
	//cout << "  New weights: " << w << endl;
  }

  return i;

}

/*
** The read_input() function reads the input files and stores the data in
** the Dvect class for use in the solution convergence algorithm.
*/
bool read_input(char **argc, Dvect &w, Dvect &y, Dvect **x, bool verbose) {
  ifstream infile;

  *x = new Dvect[20];
  for (int i=0; i<20; i++) (*x)[i].resize(10);
  w.resize(10);
  y.resize(20);

  // reading in initial weights file
  infile.open(argc[1]);
  if (infile.is_open()) {
	if (verbose) cout << "Reading in data files..." << endl;
	infile >> w;
	infile.close();
	if (verbose) cout << "Initial Weights = " << w << endl;
	  
	infile.open(argc[2]);
	if (infile.is_open()) {
	  infile >> y;
	  infile.close();
	  if (verbose) cout << "Observed y-values: " << endl << y << endl;

	  infile.open(argc[3]);
	  if (infile.is_open()) {
		for (int i=0; i<20; i++) infile >> (*x)[i];
		infile.close();
		if (verbose) cout << "Features:" << endl;
		if (verbose) for (int i=0; i<20; i++) cout << (*x)[i] << endl;
		}
	  else { cerr << "Bad input file name (x-data)." << endl; return false; }
	}
	else { cerr << "Bad input file name (y-data)." << endl; return false; }
  }
  else { cerr << "Bad input file name (weights)." << endl; return false; }
  
  return true;
}

void grad(Dvect &ret, Dvect &w, Dvect &y, Dvect *x) {
  double wTx, f;
  Dvect  a, c;

  ret.resize(w.size());
  for (int i=0; i<y.size(); i++) {
	a    = x[i] * y[i];
	wTx  = (w * x[i]).sum();
	f    = exp(wTx)/(1+exp(wTx));
	c    = x[i] * f;
	ret += (a - c);
  }

}

void gradest(Dvect &ret, Dvect &w, Dvect &y, Dvect *x) {
  double wTx, alpha, x1, x2, y1, y2, l1, l2;
  Dvect  w1, w2;

  alpha = 0.001;
  w1 = w;
  ret.resize(w.size());

  for (int i=0; i<w.size(); i++) {
	w2    = w1;
	w2[i] = w1[i]*(1-alpha);
	l1 = LL(w1,y,x);
	l2 = LL(w2,y,x);

	// calculating slope
	x1 = w1[i];
	x2 = w2[i];
	y1 = l1;
	y2 = l2;
	ret[i] = (y2 - y1)/(x2 - x1);
  }

}

/*
** Finding the next values for weights by extrapolating each element of the
** gradient to zero, and then moving the current weight in that direction
** at an input speed.  A speed of 1.0 would apply the entire extrapolation,
** while a speed of 0.5 would be half speed, and so on.
*/
void getnext(Dvect &w, Dvect &dk, double speed) {
  Dvect  wold(w);

  // each element of dk is a slope pointed toward the minimum objective
  // function value
  w = wold + dk*speed;

}

/*
** Calculate the components of the Log Liklihood (LL) objective function.
*/
void LLcomp(Dvect &l, Dvect &w, Dvect &y, Dvect *x) {
  double wTx, a, b;

  l.resize(y.size());
  for (int i=0; i<l.size(); i++) {
	wTx  = (w * x[i]).sum();
	l[i] = y[i] * wTx - log(1 + exp(wTx));
  }

}

/*
** The Log Liklihood (LL) objective function.
*/
double LL(Dvect &w, Dvect &y, Dvect *x) {
  Dvect ret;
  LLcomp(ret, w, y, x);
  return ret.sum();
}

/*
******************************************************************************
******************** Dvect CLASS DEFINITION BELOW HERE ***********************
******************************************************************************
*/

/*
** Overloading the "<<" operator allows outputing the elements to an output stream.
*/
ostream& operator<<(ostream& os, const Dvect& v) {
  os << "[ ";
  for (int i=0; i<v.size(); i++) os << v.element(i) << " ";
  os << "]";
  return os;
}

/*
** Overloading the ">>" operator allows inputing the elements from an input stream.
*/
istream& operator>>(istream& is, Dvect& v) {
  for (int i=0; i<v.size(); i++) is >> v[i];
  return is;
}

/*
** Default constructor.
*/
Dvect::Dvect(void) { 
  a  = nullptr;
  sz = 0;
}

/*
** Alternate constructor.
*/
Dvect::Dvect(int n) { 
  // setting "a" to nullptr so that the resize function does not attempt to delete it
  a = nullptr;
  // using the resize function to initialize the vector
  resize(n);
}

/*
** Copy constructor.
*/
Dvect::Dvect(const Dvect &v) {
  // setting "a" to nullptr so that the resize function does not attempt to delete it
  a = nullptr;
  // using the resize function to initialize the vector, if it works copy the input
  if (resize(v.size())) copy(v);
}

Dvect::~Dvect(void) {
  // deallocating the memory set aside for the vector
  if (a != nullptr) delete a;
}

/*
** The following inline functions are simple get/set functions.
*/
inline double Dvect::element(int i) const { return a[i]; }
inline void   Dvect::setall(double d)     { for (int i=0; i<sz; i++) a[i] = d; }
inline void   Dvect::set(int i,double d)  { if (i < sz) a[i] = d; }
inline int    Dvect::size(void) const     { return sz; }

/*
** The "*=" operator when used with two vectors multiplies each of the vectors
** together element-by-element.  This does not correspond to a true matrix multiplication.
** If the vectors are not of equal size, it does nothing.
*/
Dvect& Dvect::operator*=(const Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] *= v.a[i]; }
  return *this;
}

/*
** This version of the "*=" unary operator simply multiplies every element in the
** vector by a constant.
*/
Dvect& Dvect::operator*=(const double d) {
  for (int i=0; i<sz; i++) a[i] *= d;
  return *this;
}

/*
** This version of the "*" operator multiplies a vector by a constant.
*/
Dvect Dvect::operator*(const double d) {
  Dvect vreturn(*this);
  vreturn *= d;
  return vreturn;
}

/*
** This version of the  "*" operator multiplies two vectors together element-by-element. 
** If the vectors are not of equal size, it returns the vector on the lhs of the "*".
*/
Dvect Dvect::operator*(const Dvect &v) {
  Dvect vreturn(*this);
  vreturn *= v;
  return vreturn;
}

/*
** The "+=" operator when used with two vectors adds another vector element-by-element.
** to this one. If the vectors are not of equal size, it does nothing.
*/
Dvect& Dvect::operator+=(const Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] += v.a[i]; }
  return *this;
}

/*
** The "+" operator adds two vectors together element-by-element. If the vectors are
** not of equal size, it returns the vector on the lhs of the "+".
*/
Dvect Dvect::operator+(const Dvect &v) {
  Dvect vreturn(*this);
  vreturn += v;
  return vreturn;
}

/*
** The "-=" operator when used with two vectors subtracts another vector element-by-element.
** from this one. If the vectors are not of equal size, it does nothing.
*/
Dvect& Dvect::operator-=(const Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] -= v.a[i]; }
  return *this;
}

/*
** The "-" operator subtracts two vectors element-by-element. If the vectors are
** not of equal size, it returns the vector on the lhs of the "-".
*/
Dvect Dvect::operator-(const Dvect &v) {
  Dvect vreturn(*this);
  vreturn -= v;
  return vreturn;
}

/*
** This assignment operator uses the copy() function to copy from one vector to another
** as long as they are the same size.  Otherwise it does nothing.
*/
Dvect& Dvect::operator=(const Dvect &v) {
  resize(v.size());
  copy(v);
  return *this;
}

/*
** This assignment operator uses the setall() function to copy a double to every element
** in the vector.
*/
Dvect& Dvect::operator=(const double d) {
  setall(d);
  return *this;
}

/*
** The bracket ("[]") operator allows accessing an individual element in the vector.
*/
double Dvect::operator[](int i) const{
  if (i < sz) return a[i]; else return a[sz-1];
}

double& Dvect::operator[](int i) {
  if (i < sz) return a[i]; else return a[sz-1];
}


/*
** The resize() function resizes the vectors and destroys the data (sets to zero).
*/
bool Dvect::resize(int n) {
  // if the array is already allocated, deallocate it
  if (a != nullptr) delete a;
  // allocating a new vector ("a" for array)
  a = new double[n];
  // if the allocation was a success, the size is stored in "size"
  // otherwise, size is set to -1
  if (a != nullptr) sz = n; else sz = -1;
  // initializing the new vector with all zeroes
  for (int i=0; i<n; i++) a[i]=0;

  if (sz == -1) return false; else return true;
}

/*
** The copy() function copies the contents of one vector to another and returns "true"
** if they are the same size.  Otherwise, it does nothing and returns "false".
*/
bool Dvect::copy(const Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i]=v.a[i]; return true;  }
  else                {                                       return false; }
}

/*
** The sum() function returns a summation of all elements in a vector.
*/
double Dvect::sum(void) {
  double sum=0.0;
  for (int i=0; i<sz; i++) sum+=a[i];
  return sum;
}
