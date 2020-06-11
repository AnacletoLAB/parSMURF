// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <cinttypes>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>		// TODO: this may not work with Intel compilers. Temporary comment this line if compiling fails

class Curves {
public:
	Curves( const std::vector<uint32_t> &labels, const double * const predictions );
	Curves();
	~Curves();

	void evalAUPRCandAUROC( double * const out );
	double evalAUROC_alt();
	double evalAUROC_ok();
	double evalAUPRC();

	template <typename T, typename S>
		void apply_permutation_in_place( std::vector<T> & vec, std::vector<S> & vec2, const std::vector<std::size_t> & p );
	template <typename T>
		std::vector<T> apply_permutation( const std::vector<T>& vec, const std::vector<std::size_t>& idxes );
	template <typename T, typename S>
		void filter_dups( std::vector<T> & vec, std::vector<S> & vec2 );
	template <typename T, typename S>
		std::vector<T> cumulSum( std::vector<S> inp );
	template <typename T>
		double traps_integrate( const std::vector<T> & x, const std::vector<T> & y );
private:
	const std::vector<uint32_t> labls;
	const double * const 		preds;

	std::vector<float>			alphas;

	uint32_t					totP = 0;
	uint32_t					totN = 0;
	std::vector<double>			precision;
	std::vector<double>			recall;
	std::vector<float>			recall2;
	std::vector<float>			fpr;
	std::vector<uint32_t>		TP;
	std::vector<uint32_t>		FP;
	std::vector<uint8_t> 		tempLabs;
	std::vector<float>			tempPreds;
	std::vector<size_t>			indexes;

};

template <typename T>
void printVect( const std::vector<T> &vec ) {
	std::for_each( vec.begin(), vec.end(), [](T val) {std::cout << val << " ";} );
	std::cout << std::endl;
}

template <typename T>
void printVectC( const std::vector<T> &vec ) {
	std::for_each( vec.begin(), vec.end(), [](T val) {std::cout << (int32_t) val << " ";} );
	std::cout << std::endl;
}

template <typename T>
std::vector<T> Curves::apply_permutation( const std::vector<T>& vec, const std::vector<std::size_t>& idxes) {
    std::vector<T> sorted_vec(vec.size());
    std::transform(idxes.begin(), idxes.end(), sorted_vec.begin(), [&](std::size_t i){ return vec[i]; });
    return sorted_vec;
}

// https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of
template <typename T, typename S>
void Curves::apply_permutation_in_place( std::vector<T> & vec, std::vector<S> & vec2, const std::vector<std::size_t> & p ) {
    std::vector<bool> done( vec.size() );
    for ( std::size_t i = 0; i < vec.size(); ++i ) {
        if (done[i]) {
            continue;
        }
        done[i] = true;
        std::size_t prev_j = i;
        std::size_t j = p[i];
        while (i != j) {
            std::swap( vec[prev_j], vec[j] );
			std::swap( vec2[prev_j], vec2[j] );
            done[j] = true;
            prev_j = j;
            j = p[j];
        }
    }
}

template <typename T, typename S>
void Curves::filter_dups( std::vector<T> & vec, std::vector<S> & vec2 ) {
	size_t vecIdx = 0;				// vecIdx := result
	size_t currIdx = 0;				// currIdx := first
	size_t vecSize = vec.size();

	while(++currIdx != vecSize) {
		if(!(vec[vecIdx] == vec[currIdx]) && ++vecIdx != currIdx) {
			vec[vecIdx] = std::move( vec[currIdx] );
			vec2[vecIdx] = std::move( vec2[currIdx] );
		}
	}
	vec.resize( vecIdx + 1 );
	vec2.resize( vecIdx + 1 );
}

template <typename T, typename S>
std::vector<T> Curves::cumulSum( std::vector<S> inp ) {
	std::vector<T> tbr( inp.size(), 0 ) ;
	tbr[0] = inp[0];
	for (size_t i = 1; i < inp.size(); i++) {
		tbr[i] = tbr[i - 1] + inp[i];
	}
	return tbr;
}

template <typename T>
double Curves::traps_integrate( const std::vector<T> & x, const std::vector<T> & y ) {
	double area = 0;
	for (size_t i = 0; i < x.size() - 1; i++) {
		if (std::isnan(y[i]) | std::isnan(y[i + 1]))
			continue;
		area += ((y[i] + y[i + 1]) * (x[i + 1] - x[i] ) * 0.5);
	}
	return area;
}
