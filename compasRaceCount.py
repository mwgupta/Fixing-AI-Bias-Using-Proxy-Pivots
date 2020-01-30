import csv

with open('compas-scores-raw.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    caucasian = 0
    african_amer = 0
    hispanic = 0
    other = 0
    
    for row in csv_reader:
        race = row[8]
        if race == "Caucasian":
            caucasian += 1
        elif race == "African-American":
            african_amer += 1
        elif race == "Hispanic":
            hispanic += 1
        else:
            other += 1
            
    print("Caucasian: " + str(caucasian))
    print("African-American: " + str(african_amer))
    print("Hispanic: " + str(hispanic))
    print("Other: " + str(other))

  
