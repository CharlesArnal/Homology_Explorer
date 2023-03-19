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

// Symmetry.hh calls TriangNode.hh which calls Chirotope.hh which defines the Chirotope class (I hope as a RealChiro)
//typedef RealChiro Chirotope;


//***
// Lis le chirotope, la nouvelle triangulation, l'ancienne et ses flips, et le flip choisi dans des fichiers,
// calcule les flips de la nouvelle triangulation,
// écrit les nouveaux flips dans un fichier


int main(int argc, char *argv[])
{

std::string chiro_file = argv[1];
std::string current_triang_file = argv[2];
std::string symmetries_file = argv[3];
std::string next_triang_file = argv[4];
std::string current_flips_file = argv[5];
std::string selected_flip_file = argv[6];

std::string updated_flips_file = argv[7];

std::ifstream myfile_chiro;
std::ifstream myfile_symmetries;
std::ifstream myfile_current_triang;
std::ifstream myfile_next_triang;
std::ifstream myfile_current_flips;
std::ifstream myfile_selected_flip;

std::ofstream myfile_updated_flips;

myfile_chiro.open(chiro_file);
myfile_current_triang.open(current_triang_file);
myfile_symmetries.open(symmetries_file);
myfile_next_triang.open(next_triang_file);
myfile_current_flips.open(current_flips_file);
myfile_selected_flip.open(selected_flip_file);

myfile_updated_flips.open(updated_flips_file);


if (myfile_chiro.is_open()  && myfile_current_triang.is_open() && myfile_symmetries.is_open() 
 && myfile_next_triang.is_open() && myfile_current_flips.is_open() && myfile_selected_flip.is_open() && myfile_updated_flips.is_open())
{
 
		
  Chirotope chiro;
  chiro.read_string(myfile_chiro);
  size_type no(chiro.no());
  size_type rank(chiro.rank());
  

  //PointConfiguration points;
  //points.read(myfile_points);
  //std::cout<< points << std::endl;

  // taper TriangNode current_triang; current_triang.read(myfile_current_triang); ne semble pas marcher
  SimplicialComplex seed;
  seed.read(myfile_current_triang);
  TriangNode current_triang(0, no, rank, seed);

  SimplicialComplex next_triang_complex;
  next_triang_complex.read(myfile_next_triang);
  TriangNode next_triang(0, no, rank, next_triang_complex);

  SymmetryGroup symmetries(no);
  symmetries.read(myfile_symmetries);
  SymmetryGroup new_symmetries(symmetries, next_triang_complex);




  TriangFlips current_flips;
  current_flips.read(myfile_current_flips);
  

  


  // Comes as a collection of flips, but contains only the one good flip we want to apply
  // I do it like this to simplify reading the file, but it could be improved
  // TODO
  TriangFlips selected_flip_wrapper;
  // Needs the flip to be written as [10,3:[[{9},{3,6}]->0]]
  selected_flip_wrapper.read(myfile_selected_flip);

  Flip selected_flip;



  // this for loop has a single iteration
  for(MarkedFlips::const_iterator iter = selected_flip_wrapper.flips().begin();
       iter !=selected_flip_wrapper.flips().end(); 
       ++iter){
    const FlipRep selected_fliprep(iter->key());
    // peut-être ça à la place ? : const FlipRep current_fliprep((*iter).first);
    const Flip flip(current_triang, selected_fliprep);
    selected_flip = flip;

  }

  

  bool only_fine_triangs = false;


  TriangFlips next_flips(chiro,
            current_triang, 
            current_flips, 
            next_triang, 
            selected_flip,
            symmetries,
            new_symmetries,
            only_fine_triangs);

  myfile_updated_flips << next_flips << std::endl;
  /*
  //std::cout<< chiro << std::endl;
  std::cout<< seed << std::endl;
  //std::cout<< symmetries << std::endl;
  std::cout<< current_triang << std::endl;
  std::cout<< next_triang << std::endl;
  std::cout<< current_flips << std::endl;
  std::cout<< selected_flip << std::endl;

  //std::cout<< next_flips<< std::endl;
  */


}
else
{std::cout << "Couldn't open some files\n";}


}