1. Documentation
    a. We really need the documentation of the detail algorithm that implemented in this package, including the algorithm that calculate the two center integrals using SBT, the algorithm that comptue the position operator through the composition or Spherical Harmonics, etc.
    b. we need to add document of the interfaces of the c++ side, to enables the ease of usage of outside users.

2. python binding
    core idea, make the main object pythonic oriented. Make the user easily access to the main functions and functions relations, like composing nuemrical orbitals to construct the radial sets, or construc radial collection with radial sets, performing integral table initialization and do integrations purely in python.
    a. numerical_basis, atomic_radials, radial_collection need to be binded to python as pythonic object and support all main functions and internal data access.
    b. The pesudopotential support, we need to bind the Atom_pseudo object and the beta_radials object to python as pythonic object, and support all main functions and internal data access.
    c. We need to support the two center integrals, the op2c object as well, as a pythonic object, and allows to initialized the two center integrals table as in c++, but with python wrapped object like orbitals and pseudopotetnials that is binded before.
    

3. The numerical orbital class now is fixed, abstracting it and expose it to other kind of orbitals, like STO or GTO, by transforming the readed STO, GTO file to the numerical orbital format would make this package valuable to other orbital types.

4. Grid value sampling. Currently, we need to support the grid value sampling of the atomic orbitals, this will be important for the mesh integration of the orbitals, to computing the real space function, like \sum_ia_i\phi_i(R), or \sum_{ij}c_{ij}\phi_i(R)\phi_j(R). This will need a spatial spliting algorithm for quick neighbour list search, a k-d tree or octree would be a good choice in the future.

5. thread safty check. The op2c code base will not implementing any kind of threading, but we will need to ensure that the upper level code can use the op2c with multithreading and MPI parrallelization without worrying the thread safty issue, this can be guarded by a few test that use threading and MPI parrallelization.
