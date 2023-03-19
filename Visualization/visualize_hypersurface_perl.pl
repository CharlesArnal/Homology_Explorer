
# @ARGV should be local_path, triang_input_file_name, points_input_file_name, monomials_input_file_name, signs_input_file
#                    0                  1                         2                       3                     4


use application "tropical";

$Verbose::credits = 0;


open(INPUT, "<", "$ARGV[1]");
my $input = <INPUT>;
$input =~ s/\n//ig;
# remove the first and last {} : {{0,1,2},{0,1,3}} -> {0,1,2},{0,1,3}
$input = substr($input,1,length($input)-2);
my $triang = new Array<Set<Int>>($input);
#my $triang = new IncidenceMatrix<NonSymmetric>($input);
close(INPUT);





open(INPUT, "<", "$ARGV[2]");
my $points = new Matrix<Rational>(<INPUT>);
close(INPUT);



open(INPUT, "<", "$ARGV[3]");
my $monomials = new Matrix<Rational>(<INPUT>);
close(INPUT);


open(INPUT, "<", "$ARGV[4]");
# first line
my $signs = <INPUT>;
close(INPUT);




my $dual_sub = new fan::SubdivisionOfPoints(POINTS=>$points,MAXIMAL_CELLS=>$triang);

my $weight_vector = polytope::is_regular($points, $triang)->[1];

my $h1 = new Hypersurface<Min>(MONOMIALS=>$monomials, COEFFICIENTS=>$weight_vector);

my $h1_pw1 = $h1->PATCHWORK(SIGNS=>$signs);

my $h1_pw_r = $h1_pw1->realize();

# Works after a minute or so - must be open in Chrome.
# You can also save the file as an html with threejs($h1_pw_r->VISUAL, File=>"my_vis"); ,
# then open it with Chrome
threejs($h1_pw_r->VISUAL);
