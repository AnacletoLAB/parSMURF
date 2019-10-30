// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <cinttypes>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "ArgHandler_new.h"
#include "HyperSMURFUtils.h"

class Importer {
public:
	Importer();
	~Importer();

	static void import( const ArgHandle * const argHan, std::vector<double>& x, std::vector<uint32_t>& y, std::vector<uint32_t>& f, uint32_t * const nFolds );

	static void importParameters( std::vector<GridParams> & gridParams );
};
