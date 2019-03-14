package au.gov.nla;

import com.google.api.client.util.Key;

public class UserInfo {
    public long id;

    /*
     * The @Key annotations below map the fields from the userinfo json response from the OpenID Connect server.
     */

    @Key("iss")
    public String issuer;

    @Key("sub")
    public String subject;

    @Key("preferred_username")
    public String username;

    @Key
    public String name;

    @Key
    public String email;

    /**
     * Default constructor for JSON deserialization.
     */
    public UserInfo() {
    }

    public UserInfo(long id, String issuer, String subject, String username, String name, String email) {
        this.id = id;
        this.issuer = issuer;
        this.subject = subject;
        this.username = username;
        this.name = name;
        this.email = email;
    }
}
