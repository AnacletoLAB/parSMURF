// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "globals.h"
#include "ForestClassification.h"
#include "ForestRegression.h"
#include "ForestSurvival.h"
#include "ForestProbability.h"
#include "DataFloat.h"

using namespace ranger;

class rfRanger {
public:
	rfRanger( const uint32_t m, const bool prediction_mode, std::unique_ptr<Data> data, const uint32_t numTrees,
		uint32_t mtry, uint32_t rfThrd, uint32_t seed );
	rfRanger( Forest * inForest, const uint32_t m, const bool prediction_mode, std::unique_ptr<Data> data,
		const uint32_t numTrees, uint32_t mtry, uint32_t rfThrd, uint32_t seed );
	rfRanger( std::string forestFilename, const uint32_t m, const bool prediction_mode, std::unique_ptr<Data> data,
		const uint32_t numTrees, uint32_t mtry, uint32_t rfThrd, uint32_t seed );
	~rfRanger();

	void train( bool verbose );
	void predict( bool verbose );
	void saveForest( uint32_t nPart, std::string forestDirname );

	Forest							*	forest;
	ForestProbability				*	forestPred;
	std::string							dependent_variable_name;
	uint32_t							mtry;
	uint32_t							rfSeed;
	uint32_t							min_node_size;
	uint32_t							num_threads;
	uint32_t							numTrees;
	std::vector<double>					sample_fraction;
	double								alpha;
	double								minprop;
	bool								predict_all;
	PredictionType						prediction_type;
	uint32_t							num_random_splits;
	std::string							forestFilename;
	std::vector<std::vector<size_t>>	inbag;
	uint32_t							maxDepth;

	bool								prediction_mode;
	bool								sample_with_replacement;
	bool								memory_saving_splitting;

private:
	ImportanceMode						importance_mode;
	SplitRule							splitrule;
	std::vector<std::vector<double>>	split_select_weights;
	std::vector<std::string>			always_split_variable_names;
	std::vector<std::string>			unordered_variable_names;
	std::vector<double>					case_weights;
	std::string							status_variable_name;
	bool								keep_inbag;
	bool								holdout;
	bool								order_snp;

};
