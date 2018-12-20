# Keycloak Spring MVC/Spring Security

Based on [Vanilla Spring MVC/Spring security](../vanilla-spring-security)
     
     mvn -Djetty.port=8888 -Denv=local jetty:run
     
Admin page

     http://localhost:8888/admin
     
Welcome page
 
     http://localhost:8888/welcome
     
The contents of keycloak-local.json come from the KeyCloak admin screen.
 
Clients -> Click <Client id> -> Installation -> Format option, choose 'KeyCloak OIDC Json'
     
![Source of keycloak.json](images/keycloak-json-contents.png)
     
Refer to [KeyCloak documentation](https://www.keycloak.org/docs/latest/authorization_services/index.html#_resource_server_create_client) for how to create a client. 
