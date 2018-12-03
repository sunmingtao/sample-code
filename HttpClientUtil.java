import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.http.NameValuePair;
import org.apache.http.client.ResponseHandler;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.message.BasicNameValuePair;
import org.jboss.logging.Logger;

public class HttpClientUtil {
    
    protected static final Logger logger = Logger.getLogger(HttpClientUtil.class);
    
    public static <T> T getRequest(String uri, final ResponseHandler<? extends T> responseHandler) {
        logger.infov("URI = {0}", uri);
        try(CloseableHttpClient httpclient = HttpClients.createDefault()){
            HttpGet get = new HttpGet(uri);
            return httpclient.execute(get, responseHandler); 
        }catch(IOException e) {
            throw new RuntimeException("Post request failed.", e);
        }
    }
    
    public static <T> T postRequest(String uri, Map<String, String> params, final ResponseHandler<? extends T> responseHandler) {
        logger.infov("URI = {0}", uri);
        logger.infov("Params = {0}", params);
        try(CloseableHttpClient httpclient = HttpClients.createDefault()){
            HttpPost post = new HttpPost(uri);
            List<NameValuePair> arguments = convertToNameValuePair(params);
            post.setEntity(new UrlEncodedFormEntity(arguments));
            return httpclient.execute(post, responseHandler); 
        }catch(IOException e) {
            throw new RuntimeException("Post request failed.", e);
        }
    }
    
    private static List<NameValuePair> convertToNameValuePair(Map<String, String> params) {
        List<NameValuePair> arguments = new ArrayList<>();
        if (params != null) {
            for (String key : params.keySet()) {
                arguments.add(new BasicNameValuePair(key, params.get(key)));
            }    
        }
        return arguments;
    }
}
