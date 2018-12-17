import java.security.MessageDigest;

public final class PasswordHasher {
    public static String hash(String plainPassword, String passwordSalt) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA") ;
            md.update(passwordSalt.getBytes()) ; 
            md.update(plainPassword.getBytes()) ;
            byte[] digest = md.digest() ;
            StringBuffer sb = new StringBuffer(500) ;
            for (int i=0;i<digest.length;i++) {
                sb.append(Integer.toHexString((digest[i]&0xFF) | 0x100).substring(1,3)) ;
            }
            return sb.toString() ;
        }catch(Exception e) {
            throw new RuntimeException("Error occurred when hashing password ", e);
        }
        
    }
}
