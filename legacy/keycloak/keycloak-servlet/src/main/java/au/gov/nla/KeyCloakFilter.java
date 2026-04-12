package au.gov.nla;

import java.io.IOException;
import java.util.Arrays;

import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.google.api.client.auth.oauth2.AuthorizationCodeFlow;
import com.google.api.client.auth.oauth2.BearerToken;
import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.auth.oauth2.TokenResponse;
import com.google.api.client.http.BasicAuthentication;
import com.google.api.client.http.GenericUrl;
import com.google.api.client.http.HttpResponse;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.gson.GsonFactory;


public class KeyCloakFilter implements Filter{

    final NetHttpTransport http = new NetHttpTransport();
    final JsonFactory json = new GsonFactory();
    AuthorizationCodeFlow authFlow;
    final String serverUrl = "http://localhost:8080/auth/realms/shire/protocol/openid-connect";
    
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        authFlow = new AuthorizationCodeFlow.Builder(BearerToken.authorizationHeaderAccessMethod(),
                http, json,
                new GenericUrl(serverUrl + "/token"),
                new BasicAuthentication("servlet", "e87c7b22-3d1d-4fd9-ab01-b884609ca6db"),
                "servlet",
                serverUrl + "/auth").build();
        
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest)request;
        HttpServletResponse httpResponse = (HttpServletResponse)response;
        String servletPath = httpRequest.getServletPath();
        System.out.println("Servlet path: "+servletPath);
        if ("/login".equals(servletPath)) {
            String redirectUrl = authFlow.newAuthorizationUrl().setState("abcde").setScopes(Arrays.asList("openid")).setRedirectUri("http://localhost:8889/authcallback").build();
            System.out.println("redirectUrl " + redirectUrl);    
            Cookie cookie = new Cookie("auth_state", "abcde");
            cookie.setMaxAge(-1);
            cookie.setHttpOnly(true);
            httpResponse.addCookie(cookie);
            String backUrl = httpRequest.getParameter("backurl");
            Cookie cookie2 = new Cookie("auth_backurl", backUrl);
            cookie2.setMaxAge(-1);
            cookie2.setHttpOnly(true);
            httpResponse.addCookie(cookie2);
            httpResponse.sendRedirect(redirectUrl);
        }else if ("/authcallback".equals(servletPath)) {
            String state = httpRequest.getParameter("state");
            String authCode = httpRequest.getParameter("code");
            String cookieState = getCookie(httpRequest, "auth_state");
            if (state == null || !state.equals(cookieState)) {
                httpResponse.sendError(400, "Invalid state cookie");
            }
            TokenResponse tokenResponse = authFlow.newTokenRequest(authCode).setGrantType("authorization_code").setRedirectUri("http://localhost:8889/authcallback").execute();
            Credential credential = new Credential(BearerToken.authorizationHeaderAccessMethod()).setAccessToken(tokenResponse.getAccessToken());
            GenericUrl url = new GenericUrl(serverUrl+"/userinfo");
            HttpResponse userInfoResponse = http.createRequestFactory(credential)
                    .buildGetRequest(url)
                    .setParser(json.createJsonObjectParser())
                    .execute();
            UserInfo user = userInfoResponse.parseAs(UserInfo.class);
            String backUrl = getCookie(httpRequest, "auth_backurl");
            httpResponse.sendRedirect(backUrl);
        }else {
            chain.doFilter(request, response);
        }
    }
    
    private static String getCookie(HttpServletRequest httpRequest, String cookieName) {
        for (Cookie cookie : httpRequest.getCookies()) {
            if (cookieName.equals(cookie.getName())) {
                return cookie.getValue();
            }
        }
        return null;
    }

    @Override
    public void destroy() {
        // TODO Auto-generated method stub
        
    }
    
   
}
