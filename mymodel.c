#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sodium.h>
#include "mymodel.h"

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// functions go here