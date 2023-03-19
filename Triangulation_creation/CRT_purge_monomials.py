
import numpy as np
import os


# currently not used

def CRT_purge_monomials(relevant_monomials_file, triangulation_monomials_temp_file, triangulation_coeffs_temp_file):
  """
  Reads in relevant_monomials_file the relevant monomials, rewrites triangulation_coeffs_temp_file
  to keep only the relevant coeffs
  """
  # Loads the relevant monomials
  with open(relevant_monomials_file, 'r') as f:
    relevant_monomials =  np.loadtxt(f,dtype=int)


  
  # switch to non-projective coordinates to make the comparison possible
  relevant_monomials_non_projective = np.array([monomial[1:] for monomial in relevant_monomials])
  

  with open(triangulation_monomials_temp_file, 'r') as f:
    input_monomials =  np.loadtxt(f,dtype=int)


  with open(triangulation_coeffs_temp_file, 'r') as f:
    input_coeffs =  np.loadtxt(f,dtype=float)


  relevant_indices = [i for (i, monomial) in enumerate(input_monomials) if monomial.tolist() in relevant_monomials_non_projective.tolist()]


  #with open(os.path.join(local_path, input_triangulation_monomials_file), 'w') as f:
  #  np.savetxt(f,np.array(input_monomials[relevant_indices]),fmt='%d')

  with open(triangulation_coeffs_temp_file, 'w') as f:
    np.savetxt(f,np.array([input_coeffs[relevant_indices]]),fmt='%1.4f')


