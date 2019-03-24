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

# 4. URL:

Local (no https):
http://localhost:4567

Remote:
https://sketch2code-api.herokuapp.com

# 5. Heroku

Code is in Github repo: https://github.com/easingthemes/restrecognizer

App name: `sketch2code-api`

Heroku CI is configured for auto deploy after changes to `master` branch of Github repo.
You can also do manual deployment in Heroku admin panel.
Deployment logs are visible in Heroku admin panel.

Server logs can be accessed with:
```
heroku logs --tail --app sketch2code-api
```
More info: https://devcenter.heroku.com/articles/logging
