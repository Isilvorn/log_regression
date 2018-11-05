
#include "../include/dvect.h"

using namespace std;

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
** The default constructor simply creates an empty vector of zero size.
*/
Dvect::Dvect(void) { 
  a  = nullptr;
  sz = 0;
}

/*
** The alternate constructor creates a vector of size n with all of the elements
** set to zero.
*/
Dvect::Dvect(int n) { 
  // setting "a" to nullptr so that the resize function does not attempt to delete it
  a = nullptr;
  // using the resize function to initialize the vector
  resize(n);
}

/*
** The copy constructor creates a new vector that is exactly like the input vector,
** but occupies distinctly different memory.
*/
Dvect::Dvect(const Dvect &v) {
  // setting "a" to nullptr so that the resize function does not attempt to delete it
  a = nullptr;
  // using the resize function to initialize the vector, if it works copy the input
  if (resize(v.size())) copy(v);
}

/*
** The destructor deallocates any memory that was allocated.
*/
Dvect::~Dvect(void) {
  // deallocating the memory set aside for the vector
  if (a != nullptr) delete a;
}

/*
** The following functions are simple get/set functions.
*/
double Dvect::element(int i) const { return a[i]; }
void   Dvect::setall(double d)     { for (int i=0; i<sz; i++) a[i] = d; }
void   Dvect::set(int i,double d)  { if (i < sz) a[i] = d; }
int    Dvect::size(void) const     { return sz; }

/*
** The "*=" unary operator multiplies another vector with this one element-by-element.  
** This does not correspond to a true matrix multiplication. If the vectors are not of 
** equal size, it does nothing.
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
** If the vectors are not of equal size, it returns the vector on the lhs of the "*"
** operator.
*/
Dvect Dvect::operator*(const Dvect &v) {
  Dvect vreturn(*this);
  vreturn *= v;
  return vreturn;
}

/*
** The "+=" operator adds another vector with this one element-by-element.
** If the vectors are not of equal size, it does nothing.
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
** The "-=" operator subtracts another vector from this one element-by-element.
** If the vectors are not of equal size, it does nothing.
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
** This assignment operator uses the copy() function to copy from one vector to this one.
** This vector is resized to correspond to the input vector and then the data is copied.
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
** The bracket ("[]") operator allows accessing an individual element in the vector. If
** an index is chosen that is greater than the size of the vector, the last element in
** the vector is returned.  There is no protection for submitting a negative index; it is
** assumed that the user of this class would know that negative numbers are categorically
** illegal in this context.  The first version is the "get" version of the bracket overload,
** while the next version is the "set" version of the bracket overload.
*/
double Dvect::operator[](int i) const{
  if (i < sz) return a[i]; else return a[sz-1];
}
double& Dvect::operator[](int i) {
  if (i < sz) return a[i]; else return a[sz-1];
}


/*
** The resize() function resizes the vectors and destroys any data that might already
** exist (sets all elements to zero).
*/
bool Dvect::resize(int n) {
  // if the array is already allocated, deallocate it
  if (a != nullptr) delete a;
  // allocating a new vector ("a" for array)
  a = nullptr;
  a = new double[n];
  // if the allocation was a success, the size is stored in "size"
  // otherwise, size is set to 0 and failure is returned
  if (a != nullptr) sz = n; else { sz = 0; return false; }
  // initializing the new vector with all zeroes
  for (int i=0; i<n; i++) a[i]=0;
  // return success
  return true;
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

/*
** The exp() function takes the exponential function of every element.
*/
void Dvect::exp_elem(void) {
  for (int i=0; i<sz; i++) a[i] = exp(a[i]);
}

/*
** The apply_threshold() function sets values greater than or equal to
** the threshold to one and values less than the threshold to zero.
*/
void Dvect::apply_threshold(double d) {
  for (int i=0; i<sz; i++) a[i] = (a[i] >= d)?1.0:0.0;
}
