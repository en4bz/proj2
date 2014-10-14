import sqlite3
import csv

def main(train_inp_csv, train_outp_csv):
    db = sqlite3.connect("comp598.db")

    try:
        db.execute("create table abstracts(id int, class text, content text);")
    except sqlite3.OperationalError:
        print "Please delete the database file comp598.db and rerun the script"
        import sys
        sys.exit(1)

    with open(train_inp_csv) as inp_csv, open(train_outp_csv) as outp_csv:
        inp = csv.reader(inp_csv, delimiter=',', quotechar='"')
        outp = csv.reader(outp_csv, delimiter=',', quotechar='"')

        classes = ["cs", "stat", "physics", "math"]
        inp.next()
        outp.next()
        data = []
        for (inp_row, out_row) in zip(inp,outp):
            if out_row[1] not in classes:
                continue
            data.append((int(inp_row[0]), out_row[1], inp_row[1]))

    db.executemany('INSERT INTO abstracts VALUES (?,?,?)', data)
    db.commit()

    count = db.execute("SELECT COUNT(*) from abstracts").fetchone()[0]
    print "Database created at comp598.db!"
    print count, " entries in the database"


if __name__ == "__main__":
    import sys
    try:
        main(sys.argv[1], sys.argv[2])
    except:
        print "Unknown error"
        print "Usage: {0} <train input csv> <train output csv>".format(sys.argv[0])
        sys.exit(1)