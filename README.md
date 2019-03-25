# Get code
```
git clone https://github.com/paulochang/restrecognizer.git
```

# 1. CD into project dir
```
cd restrecognizer
```

# 2. Create package
```
mvn package
```

# 3. Start server
```
java -cp target/sketch2code-api-1.0-jar-with-dependencies.jar Main
```

# 4. Deploy to heroku
```
mvn clean heroku:deploy
```

# 5. URL:

Local (no https):
http://localhost:4567

Remote:
https://sketch2code-api.herokuapp.com