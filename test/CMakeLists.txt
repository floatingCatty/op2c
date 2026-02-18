# If GTest was fetched by root CMakeLists.txt, GTest::gtest and GTest::gtest_main targets are available.

# Helper function to add tests
function(add_op2c_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} 
        PRIVATE 
        nao_int2c 
        nao_pseudo 
        nao_core 
        nao_math 
        nao_utils 
        GTest::gtest_main
    )
    # Add MPI if available
    if(TARGET MPI::MPI_CXX)
        target_link_libraries(${test_name} PRIVATE MPI::MPI_CXX)
    endif()
    
    # Add test definition
    gtest_discover_tests(${test_name})
endfunction()

# Add existing unit test
add_op2c_test(op2c_unit_test test_op2c.cpp)

# Add new iterative test
add_op2c_test(op2c_iterative_test test_iterative.cpp)

# Add Gaunt test
add_op2c_test(op2c_gaunt_test test_gaunt.cpp)

# Add new differential test
add_op2c_test(op2c_diff_test test_diff.cpp)
