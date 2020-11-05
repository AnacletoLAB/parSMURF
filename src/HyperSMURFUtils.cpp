// parSMURF
// Alessandro Petrini, 2018-2019
#include "HyperSMURFUtils.h"

std::vector<std::string> generateRandomName( const int n ) {
	const char alphanum[] =
	        "0123456789"
	        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	        "abcdefghijklmnopqrstuvwxyz";
	std::vector<std::string> out;
	const int slen = 8;
	char stringa[slen + 1];
	stringa[slen] = 0;

	for (int i = 0; i < n; i++) {
		std::for_each( stringa, stringa + slen, [alphanum](char &c){c = alphanum[rand() % (sizeof(alphanum) - 1)];} );
		out.push_back( std::string( stringa ) );
	}

	return out;
}

void generateRandomSet( const uint32_t nn, const uint32_t mm, std::vector<double>& xx, std::vector<uint32_t>& yy, const double prob, const uint32_t seed ) {
	std::default_random_engine gen( seed );
	std::normal_distribution<> disxNeg( 0, 1 );
	std::normal_distribution<> disxPos( 5, 1 );
	std::bernoulli_distribution disy( prob );
	uint32_t cc = 0;
	std::for_each( yy.begin(), yy.end(), [disy, &gen]( uint32_t &nnn )	mutable { nnn = disy( gen ); } );
	for (size_t i = 0; i < nn; i++) {
		if (yy[i] > 0) {
			for (size_t j = 0; j < mm; j++) {
				//xx[i*(mm+1) + j] = disxNeg( gen ) + disxPos( gen );
				xx[i*(mm+1) + j] = disxPos( gen );
				//xx[i*(mm+1) + j] = i + j / (double)mm;
			}
			xx[(i+1)*(mm+1)-1] = 1.0;
		} else  {
			for (size_t j = 0; j < mm; j++) {
				xx[i*(mm+1) + j] = disxNeg( gen );
				//xx[i*(mm+1) + j] = i + j / (double)mm;
			}
			xx[(i+1)*(mm+1)-1] = 2.0;
		}
	}
}


void saveToFile( const double * const cl1, const double * const cl2, const uint32_t nn,
		const std::vector<uint32_t> * const labels, std::string outFilename ) {

	std::ofstream outFile( outFilename.c_str(), std::ios::out );
	/*
	std::for_each( cl1, cl1 + nn, [&outFile]( double nnn ) { outFile << nnn << " "; } );
	outFile << std::endl;
	std::for_each( cl2, cl2 + nn, [&outFile]( double nnn ) { outFile << nnn << " "; } );
	outFile << std::endl;
	if (!labels->empty()) {
		for_each( labels->begin(), labels->end(), [&outFile]( uint32_t nnn ) { outFile << nnn << " "; } );
		outFile << std::endl;
	}*/
	for (size_t i = 0; i < nn; i++) {
		outFile << cl1[i] << "\t" << cl2[i];
		if (!labels->empty())
			outFile << "\t" << (*labels)[i];
		outFile << std::endl;
	}
	outFile.close();
}

std::vector<std::string> split_str( std::string s, std::string delimiters ) {
	std::vector<std::string> toBeRet;
	size_t current;
	size_t next = -1;
	do {
		current = next + 1;
		next = s.find_first_of( delimiters, current );
		if (s.substr( current, next - current ) != "")
 			toBeRet.push_back( s.substr( current, next - current ) );
	} while (next != std::string::npos);
	return toBeRet;
}

void printData(const double * const xx, const uint32_t * const yy, const size_t nn, const size_t mm, const bool printLabels) {
	size_t av = 0;
	for (size_t i = 0; i < nn * (mm + 1); i++) {
		if (!(i % (mm + 1)))
			std::cout << TXT_BIYLW << i / (mm + 1) << TXT_NORML << ": ";
		std::cout << xx[i] << " ";
		if (!((i+1) % (mm + 1)) & printLabels)
			std::cout << yy[av++] << std::endl;
		else if (!((i+1) % (mm + 1)) & !printLabels)
			std::cout << std::endl;
	}
}

// TODO: This must be optimized
void transposeMatrix(double * const dst, const double * const src, const size_t nn, const size_t mm) {
	for(size_t i = 0; i < mm; i++) {
		for(size_t j = 0; j < nn; j++)
			dst[i*nn + j] = src[j*mm + i];
	}
}

void checkLoggerConfFile() {
	std::ifstream confFile( "logger.conf", std::ios::in );
	if (!confFile) {
		std::cout << TXT_BIYLW << "Logger configuration file not found (logger.conf). Creating one..." << std::endl;
		std::ofstream confFile( "logger.conf", std::ios::out );
		confFile << "* GLOBAL:" << std::endl;
		confFile << "    FORMAT               =  \"%datetime %msg\"" << std::endl;
		confFile << "    FILENAME             =  \"logGridMPI.log\"" << std::endl;
		confFile << "    ENABLED              =  true" << std::endl;
		confFile << "    TO_FILE              =  true" << std::endl;
		confFile << "    TO_STANDARD_OUTPUT   =  true" << std::endl;
		confFile << "    SUBSECOND_PRECISION  =  6" << std::endl;
		confFile << "    PERFORMANCE_TRACKING =  true" << std::endl;
		confFile << "    MAX_LOG_FILE_SIZE    =  4194304	## 4MB" << std::endl;
		confFile << "    LOG_FLUSH_THRESHOLD  =  100 ## Flush after every 100 logs" << std::endl;
		confFile << "* DEBUG:" << std::endl;
		confFile << "    FORMAT               = \"%datetime{%d/%M} %func %msg\"" << std::endl;
		confFile << "* TRACE:" << std::endl;
		confFile << "    ENABLED              =  false" << std::endl;
		confFile.close();
	} else {
		confFile.close();
	}
}

Timer::Timer() {}

double Timer::duration() {
	std::chrono::duration<double, std::ratio<1>> fp_ms = end - start;
	return fp_ms.count();
}

void Timer::startTime() {
	start = std::chrono::high_resolution_clock::now();
}

void Timer::endTime() {
	end = std::chrono::high_resolution_clock::now();
}
