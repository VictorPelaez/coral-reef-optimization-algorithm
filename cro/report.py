import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def plot_results(cro=None, filename=None):
    Fitness = cro.result["Fitness"]

    _, Ngenp1 = Fitness.shape # Ngen + 1
    Ngen = Ngenp1 - 1
    generations = np.arange(0, Ngenp1)
    Meanfitness = Fitness.mean(axis=0)
    Bestfitness = cro.opt_multiplier*((cro.opt_multiplier*Fitness).min(axis=0))

    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(True)
    ax.plot(generations, Bestfitness, "b")     
    ax.plot(generations, Meanfitness, "r--")     
    plt.xlabel('Number of generation')
    plt.xticks(generations)
    plt.xlim(-1, Ngenp1)

    plt.legend(['Best fitness' , 'Mean fitness'], loc="best")
    
    if cro:
        L = cro.L
        N = cro.N
        M = cro.M
        Fb = cro.Fb
        Fa = cro.Fa
        Fd = cro.Fd
        Pd = cro.Pd
    
        titlepro = ' Problem with Length vector (L): ' + str(L)
        titlepar = ", ".join(['Ngen: ' + str(Ngen),
                              'N: ' + str(N),
                              'M: ' + str(M),
                              'Fb: ' + str(Fb),
                              'Fa: ' + str(Fa),
                              'Fd: ' + str(Fd),
                              'Pd: ' + str(Pd)])

        plt.title(titlepro+'\n'+ titlepar)
    
    best_fitness = cro.opt_multiplier*((cro.opt_multiplier*Fitness[:, -1]).min())
    ax.scatter(Ngen, best_fitness, c='b')
    ax.annotate('Best: ' + str(best_fitness) , (1.01*Ngen, 1.01*best_fitness))
    
    if filename:
        fig.savefig(filename)
    else:
        plt.show()
