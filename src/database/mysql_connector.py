import mysql.connector
from mysql.connector import Error
print("i am here")
def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host="silu",
            user="root",
            passwd="Class@mate1234",
            database="consumer_complaint_analysis"
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection


if __name__ == "__main__":
    # Replace these with your actual database credentials
    host_name = "silu"
    user_name = "root"
    user_password = "Class@mate1234"
    db_name = "consumer_complaint_analysis"

    # Create the database connection
    connection = create_connection(host_name, user_name, user_password, db_name)