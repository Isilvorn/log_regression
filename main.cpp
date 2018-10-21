#include <iostream>
#include <fstream>
#include <string>

using namespace std;

/*
** Dvect is a class set up to store and manipulate a double precision vector.
*/
class Dvect {
public:
  Dvect(void);                  // default constructor
  Dvect(int);                   // alternate constructor
  Dvect(Dvect&);                // copy constructor
  ~Dvect();                     // destructor

  double element(int) const;    // gets the value of a specific element in the vector
  void   setall(double);        // sets all elements of a vector to the argument value
  void   set(int,double);       // sets a specific element to the argument value
  int    size(void) const;      // gets the size of the vector

  Dvect& operator*=(double);    // multiplies the vector by a constant
  Dvect& operator*=(Dvect&);    // multiplies the vector element-by-element with another
  Dvect& operator+=(Dvect&);    // adds the vector element-by-element with another
  Dvect& operator-=(Dvect&);    // subtracts another vector element-by-element from this one  
  Dvect& operator=(double);     // sets all elements of a vector to a specific value
  Dvect& operator=(Dvect&);     // sets the elements to the same as those of another
  double operator[](int) const; // allows accessing an individual element via brackets
  double& operator[](int);      // allows setting an individual element via brackets

  bool   resize(int);           // discards the data and sets the vector size to a new value
  bool   copy(Dvect&);          // copies the data from an input vector to this one
  double sum(void);             // returns the summation of all elements of this vector

  friend ostream& operator<<(ostream&,const Dvect&); // outputs all elements to a stream
  friend istream& operator>>(istream&, Dvect&);      // inputs n elements from a stream

private:
  double *a;                    // the array for the vector data
  int    sz;                    // the size of the vector
};

// read_input reads the input files and stores the data in the Dvect class
bool read_input(char **, Dvect&, Dvect&, Dvect**, bool=true);
void grad(Dvect &w, Dvect &y, Dvect *x) {
  for (int i=0; i<y.size(); i++) x[i] *= y[i];
}

int main(int argv, char **argc) {

  cout << "Executing logr with command line arguments: ";
  for (int i=0; i<argv; i++)
	cout << argc[i] << " ";
  cout << endl;

  Dvect    wvec;     // weights input file
  Dvect    yvec;     // observed y-values
  Dvect   *xvec;     // features (array size is number of data points supported)

  if (argv == 4) {
	if (read_input(argc, wvec, yvec, &xvec)) {
	  cout << "Executing grad..." << endl;
	  grad(wvec, yvec, xvec);
	  cout << "Weights = " << wvec << endl;
	  cout << "y-values: " << endl << yvec << endl;
	  cout << "Features:" << endl;
	  for (int i=0; i<20; i++) cout << xvec[i] << endl;
	  
	}
  }
  else {
	cout << "Usage:  ./logr [Initial Weights] [y-data] [x-data]" << endl;
  }

  return 0;
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
Dvect::Dvect(Dvect &v) {
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
Dvect& Dvect::operator*=(Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] *= v.a[i]; }
  return *this;
}

/*
** This version of the "*=" unary operator simply multiplies every element in the
** vector by a constant.
*/
Dvect& Dvect::operator*=(double d) {
  for (int i=0; i<sz; i++) a[i] *= d;
  return *this;
}

/*
** The "+=" operator when used with two vectors adds another vector element-by-element.
** to this one. If the vectors are not of equal size, it does nothing.
*/
Dvect& Dvect::operator+=(Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] += v.a[i]; }
  return *this;
}

/*
** The "-=" operator when used with two vectors subtracts another vector element-by-element.
** from this one. If the vectors are not of equal size, it does nothing.
*/
Dvect& Dvect::operator-=(Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] -= v.a[i]; }
  return *this;
}

/*
** This assignment operator uses the copy() function to copy from one vector to another
** as long as they are the same size.  Otherwise it does nothing.
*/
Dvect& Dvect::operator=(Dvect &v) {
  if (v.size() == sz) copy(v);
  return *this;
}

/*
** This assignment operator uses the setall() function to copy a double to every element
** in the vector.
*/
Dvect& Dvect::operator=(double d) {
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
bool Dvect::copy(Dvect &v) {
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
