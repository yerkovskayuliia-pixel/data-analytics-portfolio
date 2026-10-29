# Movie Search App (MySQL + MongoDB)

A console app that searches the **sakila** film database in MySQL and logs every search to **MongoDB**. Users can see the most popular and the most recent searches.

**Stack:** Python · PyMySQL · PyMongo · SQL

- Search by keyword, and by genre with a year range, with results paged 10 at a time
- Parameterised SQL queries (`%s`) to prevent SQL injection
- Each search is logged to MongoDB with its type, parameters, result count and timestamp
- Statistics: top-5 searches by frequency and the 5 most recent
- Errors and connection failures are handled with clear messages, so the program does not crash

## Run

Database credentials are **not** stored in the code. Copy `.env.example` to `.env` and fill in the values:

```bash
pip install pymysql pymongo python-dotenv
cp .env.example .env    # then add MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MONGO_URI
jupyter lab movie_search_sakila.ipynb
```
