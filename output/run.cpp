#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include<random>



void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


