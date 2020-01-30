import csv

with open('compas-scores-raw.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    male = 0
    female = 0
    
    for row in csv_reader:
        race = row[7]
        if race == "Male":
            male += 1
        elif race == "Female":
            female += 1
            
    print("male: " + str(male))
    print("female: " + str(female))

  
