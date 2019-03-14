package org.keycloak.adapters.springsecurity.authentication;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.keycloak.KeycloakPrincipal;
import org.keycloak.KeycloakSecurityContext;
import org.keycloak.adapters.springsecurity.account.KeycloakRole;
import org.keycloak.adapters.springsecurity.token.KeycloakAuthenticationToken;
import org.keycloak.admin.client.Keycloak;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.mapping.GrantedAuthoritiesMapper;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

public class KeycloakAuthenticationProvider2 implements AuthenticationProvider {
    private GrantedAuthoritiesMapper grantedAuthoritiesMapper;
    @Value("${env}")
    private String env;
    
    @Value("/WEB-INF/keycloak-${env}.json")
    private Resource resourceFile;

    public void setGrantedAuthoritiesMapper(GrantedAuthoritiesMapper grantedAuthoritiesMapper) {
        this.grantedAuthoritiesMapper = grantedAuthoritiesMapper;
    }

    @Override
    public Authentication authenticate(Authentication authentication) throws AuthenticationException {
        KeycloakAuthenticationToken token = (KeycloakAuthenticationToken) authentication;
        List<GrantedAuthority> grantedAuthorities = new ArrayList<GrantedAuthority>();
        for (String role : token.getAccount().getRoles()) {
            grantedAuthorities.add(new KeycloakRole(role));
        }
        Collection<?> authorities = mapAuthorities(grantedAuthorities);
        System.out.println("User roles: ");
        authorities.stream().forEach(System.out::println);
        KeycloakPrincipal<KeycloakSecurityContext> principal = (KeycloakPrincipal<KeycloakSecurityContext>)token.getAccount().getPrincipal();
        System.out.println("PreferredUserName: "+principal.getKeycloakSecurityContext().getIdToken().getPreferredUsername());
        System.out.println("Token details:"+token.getDetails());
        Keycloak kc = Keycloak.getInstance("http://localhost:8080/auth", "master", "admin", "admin", "admin-cli");
        System.out.println("User groups: ");
        kc.realms().realm("shire").users().get(principal.getName()).groups().stream().forEach(item -> System.out.println(item.getName()));
        System.out.println("Env: "+env);
        try {
            File file = resourceFile.getFile();
            ObjectMapper mapper = new ObjectMapper();
            InputStream is = new FileInputStream(file);
            KeyCloadConfigure testObj = mapper.readValue(is, KeyCloadConfigure.class);
            System.out.println("Server URL: "+testObj.authServerUrl);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return new KeycloakAuthenticationToken(token.getAccount(), token.isInteractive(), mapAuthorities(grantedAuthorities));
    }

    private Collection<? extends GrantedAuthority> mapAuthorities(
            Collection<? extends GrantedAuthority> authorities) {
        return grantedAuthoritiesMapper != null
            ? grantedAuthoritiesMapper.mapAuthorities(authorities)
            : authorities;
    }

    @Override
    public boolean supports(Class<?> aClass) {
        return KeycloakAuthenticationToken.class.isAssignableFrom(aClass);
    }
    
    @JsonIgnoreProperties(ignoreUnknown = true)
    private static class KeyCloadConfigure{
        
        @JsonProperty("realm")
        private String realm;
        
        @JsonProperty("auth-server-url")
        private String authServerUrl;

    }
}
