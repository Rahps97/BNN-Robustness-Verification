# FEM hyperparameters for the reported results

Settings that produced the FEM column, supplied by R. Singh.
`FEM_best_configurations.txt` holds the same data plus the full solution
vectors, so the energies can be verified directly with `verify_counterexamples.py`
without rerunning the solver.

| instance | energy | lr | wd | alpha | mom | c_grad | Tmin | Tmax | seed |
|---|---|---|---|---|---|---|---|---|---|
| 5x5 / 31x7x10 | -533 | 0.2356354147195816 | 0.9928606152534485 | 0.2800001800060272 | 0.997875988483429 | 4450.99658203125 | 103.00044250488281 | 287.5667724609375 | 1045 |
| 7x7 / 63x7x10 | -2643 | 0.029728282243013382 | 0.19485649466514587 | 0.18177016079425812 | 0.19137419760227203 | 176779.609375 | 12.116922378540039 | 34.266048431396484 | 386 |
| 11x11 / 127x7x10 | -8020 | 0.031124822795391083 | 5.605193857299268e-45 | 0.4876455068588257 | 5.605193857299268e-45 | 2.3926047809652573e-09 | 5.605193857299268e-45 | 3.1290994708373165e-42 | 1878 |
| 28x28 / 1023x7x10 | -1027318 | 0.007916836068034172 | 8.269223653771568e-27 | 0.2878377437591553 | 5.696139869189928e-16 | 1.5044664181118605e-08 | 1.7059775026041158e-34 | 3.929631481867403e-14 | 184 |

FEM's coordinate search is stochastic and these values are the state of that
search when the best energy was recorded, so they document what was run rather
than guaranteeing a bit-identical replay.

These runs were made with the code at the annotated tag `paper-results-v1`
(`git checkout paper-results-v1`). Later commits change how `FEM.py` generates
hyperparameter candidates and how parallel workers hold parameter state; see the
README section "The reported FEM results, and how to check them". Those changes
alter no reported number.

The 5.605193857299268e-45 entries above are the smallest positive float32
denormal. They are the signature of the multiplicative candidate rule ratcheting
a parameter downwards round after round, which later commits bound with a
relative floor. It did not prevent the search from succeeding: the 28x28 run
found the global optimum with its temperatures in that state.
