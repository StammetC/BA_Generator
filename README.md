# BA_Generator
Short program to randomly generate Büchi automata and draw them as .svg

Example Code snippet to add at the end of the code

#generates one random BA and draws it into the graphical_folder
testautomaton = generate_adjmatrix_erdosrenyi_automaton(nmin, nmax, pmin, pmax, paccmin, paccmax)
save_automata_from_data(testautomaton, "testautomaton", True)

#generates set of 'datasetsize'-many random BAs and draws them into the graphical_folder
testdataset = []
datasetsize = 10
for _ in range(datasetsize):
    testdataset.append(generate_adjmatrix_erdosrenyi_automaton(nmin, nmax, pmin, pmax, paccmin, paccmax))
save_automata_from_dataset(testdataset, "testset", True)
