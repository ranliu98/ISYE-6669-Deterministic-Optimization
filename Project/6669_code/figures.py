import matplotlib.pyplot as plt
import numpy as np

def split_data(original, position = 1):
    trial_1_linear = original.split("\n")
    trial_1_linear = [float(line.split(" ")[position]) for line in trial_1_linear]
    return trial_1_linear

def model_1_data():
    trial_1_linear = '''1.0 0.20545053482055664
1.0 0.825791597366333
1.0 3.603569507598877
0.0 17.891175270080566
1.0 149.73270988464355
0.0 1808.3734126091003'''

    trial_1_integer = '''1.0 0.20145916938781738
1.0 0.8417482376098633
1.0 3.71407151222229
1.0 20.486236095428467
1.0 155.0166072845459
1.0 2111.4954464435577'''

    x = [100, 200, 400, 800, 1600, 3200]
    x = [np.log2(i) for i in x]

    trial_1_linear = trial_1_linear.split("\n")
    #print([(line.split(" ")) for line in trial_1_linear])
    trial_1_linear = [float(line.split(" ")[1]) for line in trial_1_linear]

    trial_1_integer = trial_1_integer.split("\n")
    trial_1_integer = [float(line.split(" ")[1]) for line in trial_1_integer]

    differences = [trial_1_integer[i] - trial_1_linear[i] for i in range(len(trial_1_linear))]

    fig, ax = plt.subplots()
    ax.scatter(x, trial_1_linear, marker="x", label = "linear")
    ax.scatter(x, trial_1_integer, marker="x", alpha=0.7, label = "integer")
    for i, txt in enumerate(differences):
        ax.annotate("{:.3f}".format(txt), (x[i]-0.2, trial_1_integer[i]+40))

    plt.legend(fontsize='x-large')
    plt.ylabel("time (s)")
    plt.xlabel("log2(x)")
    #plt.savefig("model_1_one_trial.png")
    plt.show()

def model_1_data_2():
    x_1 = [100, 200, 400, 800, 1600]
    y_1 = '''0.59 18.984341382980347
0.47 79.82782006263733
0.46 353.80100655555725
0.38 1796.9403574466705
0.28 14961.767188310623'''

    x_2 = [50, 150, 300, 500, 600, 700, 900, 1000]
    y_2 = '''0.66 4.567790985107422
0.59 43.78447341918945
0.48 187.92533826828003
0.43 565.8744971752167
0.45 864.9625935554504
0.52 1252.4683060646057
0.43 2425.3777680397034
0.37 3276.1760354042053'''

    x_3 = [1100, 1200, 1300, 1400, 1500, 1700, 1800, 1900, 2000]
    y_3 = '''0.37 4372.158345460892
0.32 5728.688201189041
0.39 7382.945163726807
0.43 9406.245235204697
0.28 11806.642862081528
0.22 18062.433065891266
0.34 21951.926416873932
0.39 26531.6572971344
0.3 31731.435760974884'''

    x = x_1 + x_2 + x_3
    y_1 = [float(line.split(" ")[1]) for line in y_1.split("\n")]
    y_2 = [float(line.split(" ")[1]) for line in y_2.split("\n")]
    y_3 = [float(line.split(" ")[1]) for line in y_3.split("\n")]
    y = y_1 + y_2 + y_3

    plt.scatter(x,y, marker="x")
    plt.xlabel("value of 2n")
    plt.ylabel("time (s)")
    plt.show()
    #plt.savefig("model_1_100_trials_time.png")

def model_2_data():
    trial_1_linear_set_1 = '''1.0 0.0029916763305664062
1.0 0.018949508666992188
1.0 0.057845354080200195
1.0 0.15558433532714844
1.0 0.2702758312225342
1.0 0.45179200172424316
1.0 0.7250621318817139
1.0 1.0970673561096191
0.0 1.7812385559082031
1.0 2.353708028793335
0.0 3.686145544052124
1.0 4.59072732925415
1.0 6.693106174468994
1.0 7.630601167678833
1.0 8.406526803970337
1.0 10.552788972854614
1.0 12.987281084060669
1.0 16.236594438552856
1.0 20.386499404907227
1.0 25.025099754333496'''
    x_1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    trial_1_linear_set_2 = '''1.0 0.06582069396972656
1.0 0.4886929988861084
0.0 1.9717988967895508
1.0 4.6079137325286865
1.0 8.632996797561646
1.0 16.833163499832153
1.0 30.6201274394989
0.0 51.77075433731079
1.0 86.13691878318787
0.0 132.64377236366272
1.0 207.785475730896
1.0 264.98177433013916
1.0 350.7301836013794
0.0 450.57218956947327
1.0 561.2413566112518
1.0 704.0848052501678
0.0 870.3801231384277
0.0 1045.3600873947144
1.0 1217.2414796352386
1.0 1132.3722455501556'''
    x_2 = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570, 600]

    y_1 = split_data(trial_1_linear_set_1)
    y_2 = split_data(trial_1_linear_set_2)

    x = x_1 + x_2
    y = y_1 + y_2

    plt.scatter(x, y, marker="x", label="integer")
    plt.xlabel("2n value")
    plt.ylabel("time (s)")
    plt.savefig("model_2_time_estimate.png")
    #plt.show()

def model_2_data_2():
    x_1 = [50,100,150,200,250,300,350,400]
    y_1 = '''0.83 26.04929232597351
0.63 243.0944287776947
0.7 845.3627121448517
0.57 2512.113327741623
0.66 6263.868675708771
0.6 13283.49790430069
0.51 23482.93684029579
0.61 37319.673129320145'''

    y_1 = split_data(y_1, position=0)

    plt.scatter(x_1, y_1, marker="x")
    plt.xlabel("2n value")
    plt.ylabel("prob of integer solution (before correction)")
    plt.show()

def model_2_data_3():
    x_2 = [20,50,80,100,120,150,180,200,220,250,300,350,400]
    y_2 = '''0.97 1.870981216430664
0.87 27.567158460617065
0.79 119.56605982780457
0.72 249.68584871292114
0.8 488.63662552833557
0.7 856.0725808143616
0.68 1659.6838715076447
0.58 2525.289293050766
0.65 3776.663679122925
0.67 6384.980808734894
0.6 13418.41190958023
0.53 24124.775515794754
0.62 38528.97680521011'''
    y_2 = split_data(y_2, position=0)

    x_1 = [50, 100, 150, 200, 250, 300, 350, 400]
    y_1 = '''0.83 26.04929232597351
0.63 243.0944287776947
0.7 845.3627121448517
0.57 2512.113327741623
0.66 6263.868675708771
0.6 13283.49790430069
0.51 23482.93684029579
0.61 37319.673129320145'''
    y_1 = split_data(y_1, position=0)

    plt.scatter(x_1, y_1, marker="x", label = "before correction")
    plt.scatter(x_2, y_2, marker="x", label = "after correction")
    plt.xlabel("2n value")
    plt.ylabel("prob of integer solution")
    plt.legend()
    plt.savefig("model_2_prob.png")

model_2_data_3()