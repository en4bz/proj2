import csv


def main(output_files):
    # In case of a tie, the output with lower indexed predictions is taken
    # Thus the order is significant in this list
    # output_files:  list of paths to output files
    
    predictions = []

    for file in output_files:
        p = []
        with open(file, "rb") as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
            reader.next()
            for row in reader:
                p.append(row[1])

        predictions.append(p)

    zipped_preds = zip(*predictions)
    final_predictions = []

    categories = {}

    for prediction_group in zipped_preds:
        
        # Reset all count to zero
        for c in categories:
            categories[c] = 0

        # Count number of votes for each category
        for prediction in prediction_group:
            categories[prediction] = categories.get(prediction, 0) + 1

        max_votes = max(categories.values())

        majority_categories = []  # Category with most votes. Need not be unique

        for key, value in categories.items():
            if value == max_votes:
                majority_categories.append(key)

        # Get the first prediction in the prediction group
        # If more than is "majority", get the first one
        for prediction in prediction_group:
            if prediction in majority_categories:
                final_predictions.append(prediction)
                break


    with open("final_output.csv", "w") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "category"])
        for row in enumerate(final_predictions):
            writer.writerow(row)

    print "Output generated at final_output.csv"

if __name__ == "__main__":
    """
    Usage: python output_aggragator.py [list of output_files]
    Examples: python output_aggragator.py output_SVM.csv output_multi_NB.csv 
    """
    import sys
    main(sys.argv[1:])