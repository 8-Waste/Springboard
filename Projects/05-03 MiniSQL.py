import sqlalchemy as sa
import pandas as pd

#setup connection to MS SQL
engine = sa.create_engine('mssql+pyodbc://localhost/Country_Club?driver=SQL+Server+Native+Client+11.0')

print('\n')

#initialize two lists for questions and the corresponding SQL query to answer the question
ques = []
query = []

#populate ques and query
ques.append("""Q1: Some of the facilities charge a fee to members, but some do not.
Please list the names of the facilities that do.""")
query.append("""
SELECT Facilities.name
FROM   facilities
WHERE  membercost > 0""")

ques.append("""Q2: How many facilities do not charge a fee to members?""")
query.append("""
SELECT Count(facid) AS "NoMemberCostforFacilities"
FROM   facilities
WHERE  membercost = 0""")

ques.append("""Q3: How can you produce a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost?
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question.""")
query.append("""
SELECT facid, name, membercost, monthlymaintenance
FROM   Facilities
WHERE  membercost < ( monthlymaintenance * .2 )""")

ques.append("""Q4: How can you retrieve the details of facilities with ID 1 and 5?
Write the query without using the OR operator.""")
query.append("""
SELECT *
FROM   facilities
WHERE  facid IN( 1, 5 )""")

ques.append("""Q5: How can you produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100? Return the name and monthly maintenance of the facilities
in question.""")
query.append("""
SELECT NAME,
       monthlymaintenance,
       CASE
         WHEN monthlymaintenance < 100 THEN 'cheap'
         WHEN monthlymaintenance > 100 THEN 'expensive'
       END AS 'class'
FROM   facilities""")

ques.append("""Q6: You'd like to get the first and last name of the last member(s)
who signed up. Do not use the LIMIT clause for your solution.""")
query.append("""
SELECT firstname,
       surname
FROM   members
WHERE  joindate = (SELECT Max(joindate)
                   FROM   members)""")

ques.append("""Q7: How can you produce a list of all members who have used a tennis court?
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name.""")
query.append("""
SELECT DISTINCT Facilities.name AS court_name,
                surname + ', ' + firstname AS m_name
FROM   bookings
       INNER JOIN facilities
               ON bookings.facid = facilities.facid
       INNER JOIN members
               ON bookings.memid = members.memid
WHERE  facilities.NAME LIKE ( 'Tennis Court%' )
       AND members.memid <> 0
ORDER  BY surname + ', ' + firstname""")

ques.append("""Q8: How can you produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30? Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries.""")
query.append("""
SELECT facilities.name,
       CASE
         WHEN members.memid <> 0 THEN surname + ', ' + firstname
         WHEN members.memid = 0 THEN surname
       END AS m_name,
       CASE
         WHEN bookings.memid <> 0 THEN slots * membercost
         WHEN bookings.memid = 0 THEN slots * guestcost
       END AS 'Cost'
FROM   bookings
       INNER JOIN facilities
               ON bookings.facid = facilities.facid
       INNER JOIN members
               ON bookings.memid = members.memid
WHERE  ( starttime >= '2012-09-14 08:00:00.0000000'
         AND starttime < '2012-09-15 08:00:00.0000000' )
       AND CASE
             WHEN bookings.memid <> 0 THEN slots * membercost
             WHEN bookings.memid = 0 THEN slots * guestcost
           END > 30
ORDER  BY cost DESC""")

ques.append("""Q9: This time, produce the same result as in Q8, but using a subquery.""")
query.append("""
SELECT facilities.name,
       CASE
         WHEN members.memid <> 0 THEN surname + ', ' + firstname
         WHEN members.memid = 0 THEN surname
       END AS m_name,
       CASE
         WHEN bookings2.memid <> 0 THEN slots * membercost
         WHEN bookings2.memid = 0 THEN slots * guestcost
       END AS 'Cost'
FROM   (SELECT facid,
               memid,
               slots
        FROM   bookings
        WHERE  ( starttime >= '2012-09-14 08:00:00.0000000'
                 AND starttime < '2012-09-15 08:00:00.0000000' )) AS bookings2
       INNER JOIN facilities
               ON bookings2.facid = facilities.facid
       INNER JOIN members
               ON bookings2.memid = members.memid
WHERE  CASE
         WHEN bookings2.memid <> 0 THEN slots * membercost
         WHEN bookings2.memid = 0 THEN slots * guestcost
       END > 30
ORDER  BY cost DESC""")

ques.append("""Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members!""")
query.append("""
SELECT temp.name AS 'Facility Name',
       Round(Sum(cost), 0) AS 'Revenue'
FROM   (SELECT facilities.name,
               CASE
                 WHEN bookings.memid <> 0 THEN slots * membercost
                 WHEN bookings.memid = 0 THEN slots * guestcost
               END AS 'Cost'
        FROM   bookings
               INNER JOIN facilities
                       ON bookings.facid = facilities.facid
               INNER JOIN members
                       ON bookings.memid = members.memid) AS temp
GROUP  BY temp.name
HAVING Sum(cost) < 1000
ORDER  BY revenue DESC""")

#function to loop through questions, execute query, print result and print SQL used for result
def answer_questions(ques, query):
    print(ques,'\n')  #print the question
    result = engine.execute(query) #query the MS SQL server
    df = pd.DataFrame(result) #query result into pandas dataframe
    df.columns = result.keys() #bring in columns
    print(df.to_string(index=False),'\n') #print the dataframe (the answer to the question)
    print(query,'\n\n') #print the SQL used to query the SQL Server

if __name__ == '__main__':
    for i in range(0,len(ques)):  #run the loop for the number of questions present
        answer_questions(ques[i],query[i])

