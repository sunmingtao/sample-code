package ${package}.messagedelegate;

import javax.jms.Message;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;

import au.gov.dva.common.webservices.jaxb.JaxbXmlConverter;

/**
 * @author 
 */
public abstract class AbstractXxxResponseMessageDelegate{
	
	/** An instance of JAXB XML converter class */
	private JaxbXmlConverter jaxbXmlConverter = JaxbXmlConverter.getInstance();
	
	private final Logger logger = LoggerFactory.getLogger(getClass().getName());
	
	/**
     * Receive the message, convert the message to DTO, and process it
     * 
     * @param message The message
     */
    public void receive(Message message) {
    	//TODO
    }
}
