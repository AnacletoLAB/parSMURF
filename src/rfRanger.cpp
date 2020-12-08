// parSMURF
// Alessandro Petrini, 2018-2019
#include "rfRanger.h"

// I valori sono quelli di default in entrambi i costruttori
rfRanger::rfRanger( const uint32_t m, const bool prediction_mode, std::unique_ptr<Data> data, const uint32_t numTrees, uint32_t mtry, uint32_t rfThrd, uint32_t seed ) :
		dependent_variable_name("Labels"),
		mtry( mtry ),
		rfSeed( seed ),
		min_node_size( 0 ),
		num_threads( rfThrd ),
		numTrees( numTrees ),
		sample_fraction( std::vector<double> ({1.0}) ),
		alpha( 0.5 ),
		minprop( 0.1 ),
		predict_all( false ),
		prediction_type( RESPONSE ),
		num_random_splits( 1 ),
		forestFilename( "" ),
		maxDepth( 0 ),
		prediction_mode( prediction_mode ),
		sample_with_replacement( true ),
		memory_saving_splitting( false ),
		importance_mode( IMP_PERM_RAW ),
		splitrule( LOGRANK ),
		status_variable_name("none"),
		keep_inbag( false ),
		holdout( false ),
		order_snp( false ) {

	forest = new ForestProbability;
	forestPred = nullptr;

	if (rfSeed == 0)
		rfSeed = time( NULL );

	split_select_weights.clear();
	always_split_variable_names.clear();
	unordered_variable_names.clear();
	case_weights.clear();
	inbag.clear();

	try {
		forest->initR(dependent_variable_name, std::move(data), mtry, numTrees,
		&std::cout, rfSeed, num_threads, importance_mode, min_node_size,
		split_select_weights, always_split_variable_names,
		status_variable_name, prediction_mode, sample_with_replacement,
		unordered_variable_names, memory_saving_splitting, splitrule,
		case_weights, inbag, predict_all, keep_inbag, sample_fraction, alpha,
		minprop, holdout, prediction_type, num_random_splits, order_snp, maxDepth, forestFilename);
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << " in initR di forest" << std::endl;
	}
}

rfRanger::rfRanger( Forest * inForest, const uint32_t m, const bool prediction_mode, std::unique_ptr<Data> data, const uint32_t numTrees, uint32_t mtry, uint32_t rfThrd, uint32_t seed ) :
		dependent_variable_name("none"),
		mtry( mtry ),
		rfSeed( seed ),
		min_node_size( 0 ),
		num_threads( rfThrd ),
		numTrees( numTrees ),
		sample_fraction( std::vector<double> ({1.0}) ),
		alpha( 0.0 ),
		minprop( 0.0 ),
		predict_all( false ),
		prediction_type( RESPONSE ),
		num_random_splits( 1 ),
		forestFilename( "" ),
		maxDepth( 0 ),
		prediction_mode( prediction_mode ),
		sample_with_replacement( true ),
		memory_saving_splitting( false ),
		importance_mode( IMP_PERM_RAW ),
		splitrule( LOGRANK ),
		status_variable_name("status"),
		keep_inbag( false ),
		holdout( false ),
		order_snp( false ) {

	forestPred = new ForestProbability;
	forest = nullptr;

	if (rfSeed == 0)
		rfSeed = time( NULL );

	split_select_weights.clear();
	always_split_variable_names.clear();
	unordered_variable_names.clear();
	case_weights.clear();
	inbag.clear();

	try {
		forestPred->initR(dependent_variable_name, std::move(data), mtry, numTrees,
		&std::cout, rfSeed, num_threads, importance_mode, min_node_size,
		split_select_weights, always_split_variable_names,
		status_variable_name, prediction_mode, sample_with_replacement,
		unordered_variable_names, memory_saving_splitting, splitrule,
		case_weights, inbag, predict_all, keep_inbag, sample_fraction, alpha,
		minprop, holdout, prediction_type, num_random_splits, order_snp, maxDepth, forestFilename);
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << " in initR di forestPred" << std::endl;
	}

	try {
		size_t dependent_varID = inForest->getDependentVarId();
		std::vector<std::vector<std::vector<size_t>> > child_nodeIDs = inForest->getChildNodeIDs();
        std::vector<std::vector<size_t>> split_varIDs = inForest->getSplitVarIDs();
        std::vector<std::vector<double>> split_values = inForest->getSplitValues();
        std::vector<bool> is_ordered = inForest->getIsOrderedVariable();
		std::vector<double> class_values = ((ForestProbability*)inForest)->getClassValues();
        std::vector<std::vector<std::vector<double>>>terminal_class_counts = ((ForestProbability*)inForest)->getTerminalClassCounts();
        ((ForestProbability*) forestPred)->loadForest(dependent_varID, numTrees, child_nodeIDs, split_varIDs, split_values,
            class_values, terminal_class_counts, is_ordered);
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << " in loadForest di forestPred" << std::endl;
	}
}

rfRanger::rfRanger( std::string forestFilename, const uint32_t m, const bool prediction_mode, std::unique_ptr<Data> data,
	const uint32_t numTrees, uint32_t mtry, uint32_t rfThrd, uint32_t seed ) :
		dependent_variable_name("none"),
		mtry( mtry ),
		rfSeed( seed ),
		min_node_size( 0 ),
		num_threads( rfThrd ),
		numTrees( numTrees ),
		sample_fraction( std::vector<double> ({1.0}) ),
		alpha( 0.0 ),
		minprop( 0.0 ),
		predict_all( false ),
		prediction_type( RESPONSE ),
		num_random_splits( 1 ),
		forestFilename( forestFilename ),
		maxDepth( 0 ),
		prediction_mode( prediction_mode ),
		sample_with_replacement( true ),
		memory_saving_splitting( false ),
		importance_mode( IMP_PERM_RAW ),
		splitrule( LOGRANK ),
		status_variable_name("status"),
		keep_inbag( false ),
		holdout( false ),
		order_snp( false ) {

	forestPred = new ForestProbability;
	forest = nullptr;

	if (rfSeed == 0)
		rfSeed = time( NULL );

	split_select_weights.clear();
	always_split_variable_names.clear();
	unordered_variable_names.clear();
	case_weights.clear();
	inbag.clear();

	try {
		forestPred->initR(dependent_variable_name, std::move(data), mtry, numTrees,
		&std::cout, rfSeed, num_threads, importance_mode, min_node_size,
		split_select_weights, always_split_variable_names,
		status_variable_name, prediction_mode, sample_with_replacement,
		unordered_variable_names, memory_saving_splitting, splitrule,
		case_weights, inbag, predict_all, keep_inbag, sample_fraction, alpha,
		minprop, holdout, prediction_type, num_random_splits, order_snp, maxDepth, forestFilename);
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << " in initR di forestPred" << std::endl;
	}

}

rfRanger::~rfRanger() {
	if (forest != nullptr ) delete forest;
	if (forestPred != nullptr ) delete forestPred;
}

void rfRanger::train( bool verbose ) {
	try {
		forest->run( verbose, true );
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << " in train" << std::endl;
	}
}

void rfRanger::predict( bool verbose ) {
	try {
		forestPred->run( verbose, true );
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << " in predict" << std::endl;
	}
}

void rfRanger::saveForest( uint32_t nPart, std::string forestDirname ) {
	// What kind of format should we give to all the saved forest?
	// Pretty much something like... uhm... the user specifies a directory name
	// and the application generates one file for each partititon.
	// If we were really rad, this directory should be saved as a single
	// compressed file.
	forest->output_prefix = forestDirname + std::string( "/" ) + std::to_string( nPart ) + std::string( ".out" );

	std::cout << "saving " << forest->output_prefix << " trained forest" << std::endl;
	forest->saveToFile();
}

void rfRanger::saveImportance( uint32_t nPart, std::string forestDirname ) {
	forest->output_prefix = forestDirname + std::string( "/" ) + std::to_string( nPart ) + std::string( ".out" );
	std::cout << "saving " << forest->output_prefix << " variable importance" << std::endl;
	forest->writeImportanceFile();
}