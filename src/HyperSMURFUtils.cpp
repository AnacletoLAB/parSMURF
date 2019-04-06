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
	std::normal_distribution<> disxPos( 0, 3 );
	std::bernoulli_distribution disy( prob );
	uint32_t cc = 0;
	std::for_each( yy.begin(), yy.end(), [disy, &gen]( uint32_t &nnn )	mutable { nnn = disy( gen ); } );
	std::for_each( xx.begin(), xx.end(), [disxNeg, &gen]( double &nnn )	mutable { nnn = disxNeg( gen ); } );
	std::for_each( xx.end() - nn, xx.end(), [yy, &cc]( double &nnn )	mutable { if (yy[cc++] == 0.0) nnn = 2.0; else nnn = 1.0; } );
	for (uint32_t i = 0; i < nn; i++) {
		if (yy[i] == 1) {
			for (uint32_t j = 0; j < mm; j++) {
				xx[i + j*nn] += disxPos( gen );
			}
		}
	}
}


void saveToFile( const double * const cl1, const double * const cl2, const uint32_t nn,
		const std::vector<uint32_t> * const labels, std::string outFilename ) {

	std::ofstream outFile( outFilename.c_str(), std::ios::out );
	std::for_each( cl1, cl1 + nn, [&outFile]( double nnn ) { outFile << nnn << " "; } );
	outFile << std::endl;
	std::for_each( cl2, cl2 + nn, [&outFile]( double nnn ) { outFile << nnn << " "; } );
	outFile << std::endl;
	if (!labels->empty()) {
		for_each( labels->begin(), labels->end(), [&outFile]( uint32_t nnn ) { outFile << nnn << " "; } );
		outFile << std::endl;
	}
	outFile.close();
}

std::vector<std::string> split_str( std::string s ) {
	std::vector<std::string> toBeRet;
	std::string delimiters = " ";
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
