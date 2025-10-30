SELECT a.*, m.*
FROM public.accuracies a
JOIN public.metrics m 
  ON a.client_id = m.client_id 
 AND a.global_round = m.global_round
ORDER BY a.id ASC;