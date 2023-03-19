#include <iostream>
#include <fstream>
#include <string>

#include "CommandlineOptions.hh"
#include "ComputeTriangs.hh"
#include "SimplicialComplex.hh"
#include "RealChiro.hh"
#include "Symmetry.hh"
#include "Flip.hh"
#include "MarkedFlips.hh"
#include "TriangNode.hh"
#include "TriangFlips.hh"


// Lis la triangulation actuelle et ses flips dans un fichier,
// output les triangulations voisines et le flip qui leur donne naissance dans un fichier

int main(int argc, char *argv[])
{
     std::string chiro_file = argv[1];
     std::string current_triang_file = argv[2];
     std::string symmetries_file = argv[3];
     std::string current_flips_file = argv[4];

     std::string nb_triangs_file = argv[5];
     std::string nb_flips_file = argv[6];

     std::ifstream myfile_chiro;
     std::ifstream myfile_current_triang;
     std::ifstream myfile_symmetries;
     std::ifstream myfile_current_flips;

     std::ofstream myfile_nb_triangs;
     std::ofstream myfile_nb_flips;

     myfile_chiro.open(chiro_file);
     myfile_current_triang.open(current_triang_file);
     myfile_symmetries.open(symmetries_file);
     myfile_current_flips.open(current_flips_file);

     myfile_nb_triangs.open(nb_triangs_file);
     myfile_nb_flips.open(nb_flips_file);

     if (myfile_chiro.is_open()  && myfile_current_triang.is_open() && myfile_symmetries.is_open() 
      && myfile_nb_triangs.is_open() && myfile_current_flips.is_open() && myfile_nb_flips.is_open() )
     {
          Chirotope chiro;
          chiro.read_string(myfile_chiro);
          size_type no(chiro.no());
          size_type rank(chiro.rank());

          SimplicialComplex seed;
          seed.read(myfile_current_triang);
          TriangNode current_triang(0, no, rank, seed);
          //TriangNode current_triang;
          //current_triang.read(myfile_current_triang);

          //std::cout << current_triang <<std::endl;
          //std::cout << seed <<std::endl;

          /* // not needed for now ?
          SymmetryGroup symmetries(no);
          symmetries.read(myfile_symmetries);
          SymmetryGroup seed_symmetries(symmetries, seed);
          */

          TriangFlips current_flips;
          current_flips.read(myfile_current_flips);

          for(MarkedFlips::const_iterator iter = current_flips.flips().begin();
          iter !=current_flips.flips().end(); 
          ++iter)
          {
               // extraire un Flip au bon format de chaque flip
               FlipRep current_fliprep(iter->key()); 
               Flip flip(current_triang, current_fliprep);
               // construire la triangulation suivante en appliquant le Flip à la triangulation actuelle avec
               const TriangNode nb_triang(0, current_triang, flip);
               // calling SimplicialComplex's write function so that there is no [0->10,3 : {....}]
               nb_triang.SimplicialComplex::write(myfile_nb_triangs);
               myfile_nb_triangs <<  std::endl;
               // rmk: FlipRep hérite apparemment de Circuit
               myfile_nb_flips << current_fliprep<< std::endl;
          }
     }
     else
     {std::cout << "Couldn't open some files\n";}

    
}