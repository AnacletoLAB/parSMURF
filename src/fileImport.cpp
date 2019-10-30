// parSMURF
// Alessandro Petrini, 2018-2019
#include "fileImport.h"

void Importer::import( const ArgHandle * const argHan, std::vector<double>& x, std::vector<uint32_t>& y, std::vector<uint32_t>& f, uint32_t * const nFolds ) {

	uint32_t n;
	uint32_t con = 0;

	std::ifstream labelFile( argHan->labelFilename.c_str(), std::ios::in );

	y.clear();
	f.clear();
	x.clear();

	if (!labelFile)
		throw std::runtime_error( "Error opening label file." );

	double inDouble;
	uint32_t inUint32;
	std::cout << TXT_BIBLU << "Reading label file..." << TXT_NORML << std::endl;
	while (labelFile >> inUint32) {
		y.push_back( inUint32 );
		con++;
	}
	std::cout << TXT_BIGRN << con << " labels read" << TXT_NORML << std::endl;
	n = con;
	labelFile.close();

	con = 0;
	if (!argHan->foldFilename.empty()) {
		std::ifstream foldFile( argHan->foldFilename.c_str(), std::ios::in );
		if (!foldFile)
			throw std::runtime_error( "Error opening fold file." );

		std::cout << TXT_BIBLU << "Reading fold file..." << TXT_NORML << std::endl;
		*nFolds = 0;
		while (foldFile >> inUint32) {
			f.push_back( inUint32 );
			con++;
			if (f[con - 1] > *nFolds) *nFolds = f[con - 1];
		}
		std::cout << TXT_BIGRN << con << " values read." << TXT_NORML << std::endl;
		(*nFolds)++;
		std::cout << TXT_BIGRN << "Total number of folds: " << *nFolds << TXT_NORML << std::endl;
		foldFile.close();
		if (con != n)
			std::cout << TXT_BIRED << "WARNING: size mismatch between label and fold file!!!\033[0m" << std::endl;
	}

	// We should read and import the data from file.
	// At first, detect the number of features (columns) from data file, then import the data.
	// Labels must be appended at the end of each line

	// Import data from tsv file. Data file is supposed to be HEADERLESS
	std::ifstream dataFile( argHan->dataFilename.c_str(), std::ios::in );
	if (!dataFile)
		throw std::runtime_error( "Error opening matrix file." );

	// 1) detecting the number of columns
	std::cout << TXT_BIBLU << "Detecting the number of features from data..." << TXT_NORML << std::endl;
	// Get the length of the first line
	char c;
	while (dataFile.get(c)) {
		con++;
		if (c == '\n')
			break;
	}
	// Allocate a buffer and read the first line in its entirety
	char buffer[con];
	dataFile.seekg (0, dataFile.beg);
	dataFile.getline(buffer, con);
	// split the string according to the standard delimiters of a csv or tsv file (space, tab, comma)
	std::vector<std::string> splittedBuffer = split_str( buffer, " ,\t" );
	std::cout << TXT_BIGRN << splittedBuffer.size() << " features detected from data file." << TXT_NORML << std::endl;
	size_t tempM = splittedBuffer.size();

	con = 0;
	size_t labIdx = 0;
	std::cout << TXT_BIBLU << "Reading data file..." << TXT_NORML << std::endl;
	dataFile.seekg (0, dataFile.beg);
	while (dataFile >> inDouble) {
		x.push_back( inDouble );
		con++;
		if (!(con % tempM)) {
			if (y[labIdx++] > 0)
				x.push_back(1.0);
			else
				x.push_back(2.0);
		}
	}
	dataFile.close();

	std::cout << TXT_BIGRN << con << " values read from data file." << TXT_NORML << std::endl;

	if (con % n != 0)
		std::cout << TXT_BIRED << "WARNING: size mismatch between label and data file!!!" << TXT_NORML << std::endl;
}

void Importer::importParameters( std::vector<GridParams> & gridParams ) {
	// As now, parameter filename is hardcoded
	std::ifstream parametersFile( "params.dat", std::ios::in );
	GridParams tempPar;
	gridParams.clear();

	if (!parametersFile) {
		std::cout << "Error opening parameter files" << std::endl;
		std::exit( -1 );
	}

	std::string inStr;
	std::vector<std::string> splittedStr;
	while (std::getline(parametersFile, inStr)) {
		if (inStr.length() == 0)
			continue;
		if (inStr[0] == '#')
			continue;

		splittedStr = split_str( inStr, " " );
		// nRun, nParts, fp, ratio, k, numTrees, mtry <<-- from foldX.dat
		// becomes:
		// nFold, nParts, fp, ratio, k, numTrees, mtry <<-- expected in params.dat
		if (splittedStr.size() != 7) {
			std::cout << "error while reading params.dat" << std::endl;
			continue;
		}

		tempPar.nParts	= atoi( splittedStr[1].c_str() );
		tempPar.fp		= atoi( splittedStr[2].c_str() );
		tempPar.ratio	= atoi( splittedStr[3].c_str() );
		tempPar.k		= atoi( splittedStr[4].c_str() );
		tempPar.nTrees	= atoi( splittedStr[5].c_str() );
		tempPar.mtry	= atoi( splittedStr[6].c_str() );
		gridParams.push_back( tempPar );
	}

	parametersFile.close();
}
