package com.smt.common.logging.warning;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Log4jWarnTest {
	
	/** Logger */
	private static final Logger logger = LoggerFactory.getLogger(Log4jWarnTest.class.getName());
	
	
	@Test
	/**
	 * This is in effect:
	 * 
	 * log4j.logger.com.smt.common.logging.warning=WARN
	 * 
	 */
	public void testLogPackage() {
		// TRACE, DEBUG, and INFO are not logged
		logger.trace("Trace");
		logger.debug("Debug");
		logger.info("Info");
		//Only above WARN are ogged
		logger.warn("Warning");
		logger.error("Error");
	}
}
