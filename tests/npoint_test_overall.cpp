/**
 * npoint_test_overall.cpp
 * @author Bill March (march@gatech.edu)
 *
 * Mega-test for all npoint stuff.
 * This is the tests against ground truth.
 * 
 * 2 sources of ground truth: 
 * 1) examples I can work out by hand
 * 2) Results from the auton code and ntropy - I think these were both 
 * extensively validated.
 */

// apparently this needs to go before we include unit test stuff
#define BOOST_TEST_MODULE npoint_overall_test
#include <boost/test/unit_test.hpp>


