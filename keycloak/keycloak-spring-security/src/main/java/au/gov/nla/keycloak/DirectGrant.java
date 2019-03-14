package au.gov.nla.keycloak;

import java.io.IOException;
import java.text.MessageFormat;

import javax.ws.rs.client.ClientRequestContext;
import javax.ws.rs.client.ClientRequestFilter;

import org.jboss.resteasy.client.jaxrs.ResteasyClientBuilder;
import org.keycloak.OAuth2Constants;
import org.keycloak.admin.client.Keycloak;
import org.keycloak.admin.client.KeycloakBuilder;
import org.keycloak.authorization.client.AuthzClient;
import org.keycloak.authorization.client.util.HttpResponseException;

public class DirectGrant {
    
    private static final String LOCAL_USER = "7463098a-5218-4329-a5cd-c4f32db18a58";
    private static final String DEV_USER = "c6b6c732-3c42-47c0-aa07-2d6f6b6ea6ca";
    

    public static void main(String[] args) throws Exception{
        
        System.out.println(MessageFormat.format("<p>abcde''s</p>", new Object[] {}));
//        String accessTokenString = login("smt3", "1234");
//        System.out.println("User id="+getUserId(accessTokenString));
//        accessTokenString = login("smt3", "123");
//        if (accessTokenString == null) {
//            System.out.println("Invalid username or password");
//        }
//        String userId = DEV_USER;
//        Keycloak kc = getKeyCloakInstance2();
//        RealmResource realmResource = kc.realm("trove");
//        String token = kc.tokenManager().getAccessTokenString();
//        System.out.println(token);
//        System.out.println(realmResource.groups().groups());
       // System.out.println(realmResource.groups().groups());
//        System.out.println(usersResource.get(userId).roles().getAll().getRealmMappings());
//        System.out.println(usersResource.get(userId).toRepresentation().getFirstName());
//        UserRepresentation user = usersResource.get(userId).toRepresentation();
//        //user.singleAttribute("test_name", "test value");
//        //user.setEmail("smt222@gmail.com");
//        usersResource.get(userId).update(user);
//        
//        
//        UserResource userResource = usersResource.get(userId);
//        List<RoleRepresentation> roles = userResource.roles().getAll().getRealmMappings();
//        System.out.println(roles);
//        RoleRepresentation newRole = realmResource.roles().list().get(2);
//        roles.add(newRole);
//        System.out.println("New role "+newRole);
//        System.out.println(roles);
//        System.out.println("Before: "+usersResource.get(userId).roles().getAll().getRealmMappings());
//        RoleScopeResource roleScope = userResource.roles().realmLevel();
//        roleScope.add(Arrays.asList(newRole));
//        System.out.println("Latest: "+usersResource.get(userId).roles().getAll().getRealmMappings());
        
    }
    
    private static String login(String username, String password) {
        try {
            AuthzClient authzClient = AuthzClient.create();
            String accessTokenString = authzClient.obtainAccessToken(username, password).getToken();
            return accessTokenString;
        }catch(HttpResponseException e) {
            if (e.getStatusCode() == 401) {
                return null;
            }else {
                throw e;
            }
        }
    }
    
//    private static String getUserId(String accessTokenString) throws Exception{
//        InputStream configStream = Thread.currentThread().getContextClassLoader().getResourceAsStream("keycloak.json");
//        KeycloakDeployment deployment = KeycloakDeploymentBuilder.build(configStream);
//        TokenVerifier<AccessToken> tokenVerifier = AdapterTokenVerifier.createVerifier(accessTokenString, deployment, false, AccessToken.class);
//        AccessToken accessToken = tokenVerifier.verify().getToken();
//        return accessToken.getSubject();
//    }

    
    private static Keycloak getKeyCloakInstance0() {
        return Keycloak.getInstance( "http://localhost:8080/auth", "master", "admin", "admin", "security-admin-console"); 
    }
    
    private static Keycloak getKeyCloakInstance() {
        return KeycloakBuilder.builder().serverUrl("https://login-devel.nla.gov.au/auth")
                .realm("trove").grantType(OAuth2Constants.CLIENT_CREDENTIALS).clientId("trove")
                .clientSecret("465885a3-94dc-47c9-8765-6a88433d4aef").resteasyClient(new ResteasyClientBuilder().connectionPoolSize(10).build().register(new LoggingFilter())).build();
    }
    
    private static Keycloak getKeyCloakInstance3() {
        return KeycloakBuilder.builder().serverUrl("https://login-devel.nla.gov.au/auth")
                .realm("shire").grantType(OAuth2Constants.CLIENT_CREDENTIALS).clientId("sprightly")
                .clientSecret("185d1288-c1d5-4ee6-8449-830c985eb179").resteasyClient(new ResteasyClientBuilder().connectionPoolSize(10).build().register(new LoggingFilter())).build();
    }
    
    
    private static Keycloak getKeyCloakInstance2() {
        return KeycloakBuilder.builder().serverUrl("http://localhost:8080/auth")
                .realm("trove").grantType(OAuth2Constants.CLIENT_CREDENTIALS).clientId("trove")
                .clientSecret("16e46339-53e8-4eea-b3c0-0e72f282f3d3").resteasyClient(new ResteasyClientBuilder().connectionPoolSize(10).build().register(new LoggingFilter())).build();
    }
    
    private static class LoggingFilter implements ClientRequestFilter {

        @Override
        public void filter(final ClientRequestContext requestContext) throws IOException {
            if (requestContext != null) {
                System.out.println("Request context URI: "+requestContext.getUri());
                System.out.println("Headers: " + requestContext.getHeaders());
                System.out.println("Entity: " + requestContext.getEntity());
                System.out.println("Entity class " + requestContext.getEntityClass());
                if (requestContext.getEntity() != null) {
                    System.out.println("Entity " +requestContext.getEntity().getClass());
                }
                
            }
        }
    }
    
}
