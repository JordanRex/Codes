setwd("~/R/neo4j")

# libraries
{
  library(dplyr)
  library(data.table)
  library(magrittr)
  library(stringr)
}

# reading the files
{
  x = fread('Neo4j_input1.csv', header = T, stringsAsFactors = F)

  y = fread('Neo4j_input2.csv', header = T, stringsAsFactors = F, check.names = T)

  z1 = readxl::read_xlsx('Seq_606_MOBO.xlsx', sheet = 1)
  z2 = readxl::read_xlsx('Seq_606_MOBO.xlsx', sheet = 2) %>%
    rename_(.dots = setNames(colnames(.), c("QUESTION", "Q_ID")))
}

# processing
{
  # not %in% function
  '%!in%' <- function(x,y)!('%in%'(x,y))

  # adding a dummy dispatch flag
  y %<>%
    rowwise() %>%
    mutate(DISPATCH_FLAG = if_else(AUTO_LOG_RECOMMENDS == "Empty", as.integer(0), sample(0:1, 1, prob = c(0.25, 0.75), replace = T)),
           REP_DISPATCH_FLAG = if_else(DISPATCH_FLAG == 1, sample(0:1, 1, prob = c(0.01, 0.99)), as.integer(0)))

  x1 = x %>%
    select(SVC_ACTVY_ID, QUESTION = Question_x, ANS_RAW = Answer, ANS_CLEAN = Answer_n, ARTICLE_ID = AUTO_LOG_ARTICLE_ID, PATH = Tag) %>%
    left_join(z2) %>%
    filter(is.na(Q_ID) == F, ARTICLE_ID == "SLN297606")

  y1 = y %>%
    select(YEAR_WEEK, SVC_ACTVY_ID, SR_ID, ASSET_ID = ASST_ID, CHANNEL = ACTVY_TYPE_CD, AGENT_MNGR_ID = SR_OWNER_ASSOC_BDGE_NBR, AGENT_ID = CRT_ASSOC_BDGE_NBR, CALL_DUR = Call_Time, PRODUCT_DESC, LOCATION = ASSOC_LOC_NM, IS_REPEAT_FLG, HAS_REPEAT_FLG, ARTICLE_ID = AUTO_LOG_ARTICLE_ID, ARTICLE_TITLE = AUTO_LOG_ARTICLE_TITLE, RECOMMENDATION = AUTO_LOG_RECOMMENDS, DISPATCH_FLAG, REP_DISPATCH_FLAG) %>%
    filter(ARTICLE_ID == "SLN297606")

  xy = left_join(x1, y1) %>%
    mutate(RECOMMENDATION = if_else(RECOMMENDATION %in% c("Empty", "SLN300599"), "None", RECOMMENDATION)) %>%
    mutate(ANS_CLEAN = if_else(str_detect(pattern = "UNABLE TO REMOVE CRUS", ANS_CLEAN), "UNABLE TO REMOVE CRUS", ANS_CLEAN),
           ANS_CLEAN = if_else(str_detect(pattern = "POWER BUTTON SEEMS TO BE CAUSING ISSUE", ANS_CLEAN), "POWER BUTTON SEEMS TO BE CAUSING ISSUE", ANS_CLEAN),
           ANS_CLEAN = if_else(ANS_CLEAN %in% c("YES", "MOTHERBOARD.", "DAUGHTERBOARD.", "DC-IN CABLE"), "YES", "NO"),
           PATH = str_replace(Q_ID, pattern = "Q", replacement = ""))

  xy1 = xy %>%
    select(RECOMMENDATION, SVC_ACTVY_ID, DISPATCH_FLAG, REP_DISPATCH_FLAG) %>%
    group_by(RECOMMENDATION) %>%
    summarise(NO_ACTVY = n_distinct(SVC_ACTVY_ID),
              TOTAL_DISPATCH = sum(DISPATCH_FLAG),
              TOTAL_REP_DISPATCH = sum(REP_DISPATCH_FLAG)) %>%
    ungroup() %>%
    mutate(R_ID = paste0("R",dense_rank(RECOMMENDATION)),
           R_ACTVY_PERC = (NO_ACTVY*100)/sum(NO_ACTVY)) %>%
    rowwise() %>%
    mutate(AVG_CUST_SAT = if_else(RECOMMENDATION == "None", as.integer(2), sample(4:8, 1)))

  xy %<>%
    group_by(SVC_ACTVY_ID) %>%
    mutate(Q_NEXT = lead(Q_ID, n = 1)) %>%
    ungroup() %>%
    left_join(xy1) %>%
    mutate(Q_NEXT = if_else(is.na(Q_NEXT) == T, R_ID, Q_NEXT))

  Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }

  xy2 = xy %>%
    group_by(Q_ID, Q_NEXT, ANS_CLEAN, RECOMMENDATION) %>%
    summarise(USUAL_POS_PATH = Mode(PATH),
              AVG_CALL_DUR = if_else(is.nan(mean(as.numeric(CALL_DUR), na.rm = T)) == T,
                                     0,
                                     mean(as.numeric(CALL_DUR), na.rm = T)),
              RULE_DIS_ACTIVITIES = n_distinct(SVC_ACTVY_ID),
              TOTAL_DISPATCH = sum(DISPATCH_FLAG),
              TOTAL_REP_DISPATCH = sum(REP_DISPATCH_FLAG)) %>%
    ungroup() %>%
    rowwise() %>%
    mutate(AVG_CUST_SAT = if_else(RECOMMENDATION == "None", sample(1:5, 1), sample(4:10, 1)),
           AVG_COST_HANDLING = round((((TOTAL_DISPATCH * sample(130:150, 1)) + ((RULE_DIS_ACTIVITIES - TOTAL_DISPATCH) * sample(20:30, 1)))/RULE_DIS_ACTIVITIES), digits = 0),
           AVG_COST_PATH_PER_ACTVY = runif(1, 0.3, 0.9))

  # to find parent questions
  Q_ID = xy$Q_ID %>% unique()
  Q_NEXT = xy$Q_NEXT %>% unique()
  Q_PARENT = setdiff(Q_ID, Q_NEXT)
  # Q_PARENT = Q1
  # the only parent node is Q1
}

# cypher language queries
{
  ## cypher1
  cypher1 = xy %>%
    group_by(Q_ID) %>%
    summarise(NODE_DIS_ACTIVITIES = n_distinct(SVC_ACTVY_ID)) %>%
    left_join(xy[, c("Q_ID", "QUESTION", "ARTICLE_ID", "ARTICLE_TITLE")]) %>%
    ungroup() %>%
    select(Q_ID, NODE_DIS_ACTIVITIES, QUESTION, ARTICLE_ID, ARTICLE_TITLE) %>%
    distinct() %>%
    mutate(CYPHER1 = paste0('CREATE (', Q_ID, ':QUESTION {DESCP:"', QUESTION, '", `ARTICLE ID`: "', ARTICLE_ID, '", `ARTICLE TITLE`: "', ARTICLE_TITLE, '", `NUMBER OF ACTVY`: ', NODE_DIS_ACTIVITIES, '})')) %>%
    select(CYPHER1)

  ## cypher2
  cypher2 = xy1 %>%
    mutate(CYPHER2 = paste0('CREATE (', R_ID, ':RECOMMENDATION {DESCP:"', RECOMMENDATION, '", `NUMBER OF ACTVY`: ', NO_ACTVY, ', `NUMBER OF DISPATCH`: ', TOTAL_DISPATCH, ', `NUMBER OF REP DISPATCH`: ', TOTAL_REP_DISPATCH, ', `AVG CUST SAT`: ', AVG_CUST_SAT, '})')) %>%
    select(CYPHER2)

  ## cypher3
  cypher3 = xy2 %>%
    mutate(CYPHER3 = paste0('CREATE (', Q_ID, ')- [:', ANS_CLEAN, ' { RECOMMENDATION: "', RECOMMENDATION, '", `USUAL PATH LOCATION`: ', USUAL_POS_PATH, ', `AVG CALL DUR`: ', round(AVG_CALL_DUR, digits = 0), ', `NUMBER OF ACTVY`: ', RULE_DIS_ACTIVITIES, ', `AVG COST PER ACTVY`: ', round(AVG_COST_PATH_PER_ACTVY, digits = 1), ', `AVG COST HANDLING`: ', AVG_COST_HANDLING, ', `AVG CUST SAT`: ', AVG_CUST_SAT, '}] -> (', Q_NEXT, ')')) %>%
    select(CYPHER3)

  ## cypher4
  cypher4 = xy %>%
    select(ARTICLE_ID, ARTICLE_TITLE, SVC_ACTVY_ID) %>%
    group_by(ARTICLE_ID, ARTICLE_TITLE) %>%
    summarise(TOTAL_ACTVY = n_distinct(SVC_ACTVY_ID)) %>%
    ungroup() %>%
    mutate(CYPHER4 = paste0('CREATE (`ISSUE 1`:ISSUE {`ARTICLE TITLE`: "', ARTICLE_TITLE, '", `ARTICLE ID`: "', ARTICLE_ID, '", `TOTAL ACTVY`: ', TOTAL_ACTVY, '})')) %>%
    select(CYPHER4) %>%
    rbind(paste0('CREATE (`ISSUE 1`)- [:QUEST {DESCP: "AFTER DIAGNOSIS"}] -> (Q1)'))

  ## final cypher
  cypher = bind_rows(list(cypher1, cypher2, cypher3, cypher4))
  cypher_full = data.frame(CYPHER = na.omit(unlist(cypher)), row.names = NULL)

  cypher = cbind(cypher, cypher_full)

  write.csv(cypher, 'cypher.csv')
}
