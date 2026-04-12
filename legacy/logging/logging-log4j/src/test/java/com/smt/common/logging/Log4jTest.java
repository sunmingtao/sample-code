package com.smt.common.logging;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Log4jTest {
	
	/** Logger */
	private static final Logger logger = LoggerFactory.getLogger(Log4jTest.class.getName());
	
	@Test
	public void test(){
		LoggingInterface logging = new LoggingInterface();
		logging.log();
	}
	
	@Test
	/**
	 * Error also logs in example.log
	 * 
	 * Following two lines are in effect
	 * 
	 * log4j.appender.R.threshold=error
	 * log4j.appender.R.File=example.log
	 */
	public void testLogError(){
		logger.error("This is an error");
		//Warning is not logged in example.log
		logger.warn("This is an warning");
	}
	
	@Test
	/**
	 * This is in effect:
	 * 
	 * log4j.rootLogger=DEBUG
	 */
	public void testDebugAbove(){
		//Trace is not logged
		logger.trace("Trace");
		//Above Debug are all logged
		logger.debug("Debug");
		logger.info("Info");
		logger.warn("Warning");
		logger.error("Error");
	}
	
	
}
