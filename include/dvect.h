
#ifndef DVECT_H
#define DVECT_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
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
  void   exp_elem(void);           // takes the exponential function of every element
  void   apply_threshold(double);  // sets values >= threshold to 1 and < threshold to 0

  friend ostream& operator<<(ostream&,const Dvect&); // outputs all elements to a stream
  friend istream& operator>>(istream&, Dvect&);      // inputs n elements from a stream

private:
  double *a;                       // the array for the vector data
  int     sz;                      // the size of the vector
};

#endif // DVECT_H

