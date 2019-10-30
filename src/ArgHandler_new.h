// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <string>
#include <cinttypes>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <time.h>
#include <json.hpp>
#include "HyperSMURFUtils.h"

class ArgHandle {
public:
	ArgHandle( int argc, char **argv, std::vector<GridParams> &gridParams );
	virtual ~ArgHandle();

	void processCommandLine( int rank );
	void processMtry( uint32_t mm );
	void printConfig( uint32_t n, uint32_t m );
	void fillParams( jsoncons::json * params, std::vector<GridParams> &gridParams );

	std::vector<GridParams> &gridParams;

	std::string		dataFilename;
	std::string		foldFilename;
	std::string		labelFilename;
	std::string		outFilename;
	std::string		forestDirname;
	std::string		timeFilename;
	std::string		extConfigFilename;

	uint32_t		m;
	uint32_t		n;
	double			prob;
	uint32_t		nFolds;
	uint32_t		seed;
	uint32_t		verboseLevel;
	uint32_t		ensThreads;
	uint32_t		rfThreads;
	uint8_t			wmode;
	uint8_t			woptimiz;

	bool			generateRandomFold;
	bool			readNFromFile;
	bool			simulate;
	bool			verboseMPI;
	bool			noMtSender;
	bool			externalConfig;
	bool			printCurrentConfig;

	uint32_t		minFold;
	uint32_t		maxFold;

protected:
	void displayHelp();
	void printLogo();
	void checkConfig( int rank );
	void checkCommonConfig( int rank );
	void jsonImport( std::string cfgFilename );

	template <typename T>
	T getFromJson( jsoncons::json * jStrct, std::string field, T currValue ) {
		if (!(jStrct->contains(field)))
			return currValue;

		T toBeReturned;
		bool okay = false;
		try {
			toBeReturned = jStrct->get_with_default( field ).as<T>();
			okay = true;
		} catch (const std::exception& e) {
			std::cout << "Error while getting " << field << " from json: " << e.what() << std::endl;
		}
		if (okay) {
			return toBeReturned;
		}
		else {
			return currValue;
		}
	}

	int					argc;
	char			**	argv;
	jsoncons::json		jsCfg;
	std::string			mode;
	std::string			optim;

};
