
# Could probably be optimized a bit
# @ARGV should be output triang file name, all triangulation monomials, triangulation coeffs
#                              0                          1                        2           

use application "tropical";

$Verbose::credits = 0;

open(INPUT, "<", "$ARGV[1]");
my $all_monomials = new Matrix<Rational>(<INPUT>);
close(INPUT);

my @all_coeffs =();
open(INPUT, "<", "$ARGV[2]");
while(<INPUT>){
@all_coeffs = $_ ;
}
close(INPUT);

my $h1 = new Hypersurface<Min>(MONOMIALS=>$all_monomials, COEFFICIENTS=>@all_coeffs);

open(my $f, ">", "$ARGV[0]");
my $triang = $h1->DUAL_SUBDIVISION->MAXIMAL_CELLS;
$triang =~ s/\n//ig;
$triang = "{" . $triang . "}";
print $f $triang;
close $f;

