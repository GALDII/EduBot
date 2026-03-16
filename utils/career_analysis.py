import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm import get_llm
from langchain_core.messages import HumanMessage
from utils.web_search import web_search


def calculate_career_readiness_score(
    resume_text: str,
    marks_df: pd.DataFrame,
    target_role: str,
    job_requirements: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate career readiness score based on resume, marks, and job requirements.
    Returns score (0-100) with breakdown.
    """
    try:
        llm = get_llm()
        
        # Extract skills from resume
        skills_prompt = f"""Extract all technical skills, programming languages, tools, and technologies mentioned in this resume:
{resume_text[:2000]}

Return a JSON object with:
- "skills": list of all technical skills found
- "experience_years": estimated years of experience
- "education_level": highest education level
- "projects": number of projects mentioned
"""
        
        skills_data = llm.invoke([HumanMessage(content=skills_prompt)]).content
        
        # Calculate academic performance
        academic_score = 0
        if marks_df is not None and not marks_df.empty:
            numeric_cols = marks_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                avg_marks = marks_df[numeric_cols].mean().mean()
                academic_score = min(100, (avg_marks / 100) * 30)  # Max 30 points
        
        # Get job requirements if not provided
        if not job_requirements and target_role:
            search_query = f"{target_role} job requirements skills qualifications"
            web_results = web_search(search_query)
            if web_results:
                job_requirements = "\n".join([r.get("content", "")[:500] for r in web_results[:3]])
        
        # Calculate match score
        match_prompt = f"""Compare this candidate profile against the job requirements and calculate a readiness score.

CANDIDATE PROFILE:
{skills_data}

JOB REQUIREMENTS:
{job_requirements or "General requirements for " + target_role}

Return a JSON object with:
- "overall_score": number 0-100
- "skills_match": number 0-100 (how well skills match)
- "experience_match": number 0-100
- "education_match": number 0-100
- "gaps": list of missing skills or qualifications
- "strengths": list of strong points
"""
        
        match_result = llm.invoke([HumanMessage(content=match_prompt)]).content
        
        # Parse LLM response (simplified - in production, use proper JSON parsing)
        import json
        import re
        match_json = re.search(r'\{.*\}', match_result, re.DOTALL)
        if match_json:
            try:
                match_data = json.loads(match_json.group())
            except:
                match_data = {}
        else:
            match_data = {}
        
        # Calculate final score
        skills_score = match_data.get("skills_match", 50)
        experience_score = match_data.get("experience_match", 50)
        education_score = match_data.get("education_match", 50)
        
        # Weighted average
        overall = (
            academic_score * 0.2 +
            skills_score * 0.4 +
            experience_score * 0.25 +
            education_score * 0.15
        )
        
        return {
            "overall_score": round(overall, 1),
            "academic_score": round(academic_score, 1),
            "skills_match": round(skills_score, 1),
            "experience_match": round(experience_score, 1),
            "education_match": round(education_score, 1),
            "gaps": match_data.get("gaps", []),
            "strengths": match_data.get("strengths", []),
            "target_role": target_role
        }
    
    except Exception as e:
        return {
            "overall_score": 0,
            "error": str(e),
            "academic_score": 0,
            "skills_match": 0,
            "experience_match": 0,
            "education_match": 0,
            "gaps": [],
            "strengths": []
        }


def analyze_skill_gaps(
    resume_text: str,
    target_role: str,
    job_requirements: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze skill gaps between user and target role."""
    try:
        llm = get_llm()
        
        # Extract user skills
        skills_prompt = f"""Extract all skills, technologies, and competencies from this resume:
{resume_text[:2000]}

Return a JSON array of skills: ["skill1", "skill2", ...]
"""
        user_skills_text = llm.invoke([HumanMessage(content=skills_prompt)]).content
        
        # Get required skills for role
        if not job_requirements:
            search_query = f"{target_role} required skills technologies tools"
            web_results = web_search(search_query)
            if web_results:
                job_requirements = "\n".join([r.get("content", "")[:500] for r in web_results[:3]])
        
        gap_prompt = f"""Compare user skills against job requirements and identify gaps.

USER SKILLS:
{user_skills_text}

JOB REQUIREMENTS:
{job_requirements or f"Standard requirements for {target_role}"}

Return JSON:
{{
  "user_skills": ["skill1", "skill2"],
  "required_skills": ["skill1", "skill2"],
  "matched_skills": ["skill1"],
  "missing_skills": ["skill2"],
  "skill_gaps": [
    {{"skill": "Python", "priority": "high", "reason": "Required for data analysis"}}
  ]
}}
"""
        
        gap_result = llm.invoke([HumanMessage(content=gap_prompt)]).content
        
        import json
        import re
        gap_json = re.search(r'\{.*\}', gap_result, re.DOTALL)
        if gap_json:
            try:
                return json.loads(gap_json.group())
            except:
                pass
        
        return {
            "user_skills": [],
            "required_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "skill_gaps": []
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "user_skills": [],
            "required_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "skill_gaps": []
        }


def analyze_performance_trends(marks_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze semester-wise performance trends and predict future performance."""
    try:
        if marks_df is None or marks_df.empty:
            return {"error": "No data available"}
        
        # Find semester column
        semester_col = None
        for col in marks_df.columns:
            if 'semester' in col.lower() or 'sem' in col.lower():
                semester_col = col
                break
        
        if not semester_col:
            # Try to infer from data
            numeric_cols = marks_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric data found"}
        
        # Calculate trends
        numeric_cols = marks_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"error": "No numeric columns found"}
        
        # Group by semester if available
        if semester_col:
            trends = marks_df.groupby(semester_col)[numeric_cols].mean()
        else:
            # Use row index as semester
            trends = marks_df[numeric_cols].mean(axis=1)
        
        # Calculate average per semester
        if isinstance(trends, pd.Series):
            semester_avg = trends.values
        else:
            semester_avg = trends.mean(axis=1).values
        
        # Simple linear prediction for next semester
        if len(semester_avg) >= 2:
            x = np.arange(len(semester_avg))
            coeffs = np.polyfit(x, semester_avg, 1)
            next_semester_pred = np.polyval(coeffs, len(semester_avg))
            trend_direction = "increasing" if coeffs[0] > 0 else "decreasing"
        else:
            next_semester_pred = semester_avg[-1] if len(semester_avg) > 0 else 0
            trend_direction = "stable"
        
        return {
            "semester_averages": semester_avg.tolist(),
            "trend_direction": trend_direction,
            "next_semester_prediction": round(float(next_semester_pred), 2),
            "overall_average": round(float(np.mean(semester_avg)), 2),
            "best_semester": int(np.argmax(semester_avg)) + 1,
            "improvement_rate": round(float(coeffs[0]) if len(semester_avg) >= 2 else 0, 2)
        }
    
    except Exception as e:
        return {"error": str(e)}


def get_recommendations(
    resume_text: str,
    marks_df: pd.DataFrame,
    target_role: str,
    skill_gaps: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate personalized recommendations for improvement."""
    try:
        llm = get_llm()
        
        gaps_text = "\n".join([f"- {g.get('skill', '')}: {g.get('reason', '')}" for g in skill_gaps[:10]])
        
        prompt = f"""Based on this candidate profile and identified skill gaps, provide specific, actionable recommendations.

CANDIDATE PROFILE:
Resume: {resume_text[:1500]}
Target Role: {target_role}

SKILL GAPS:
{gaps_text}

Return JSON array of recommendations:
[
  {{
    "type": "course|certification|project|skill",
    "title": "Recommendation title",
    "description": "Why this helps",
    "priority": "high|medium|low",
    "resource": "Link or name of resource"
  }}
]
"""
        
        result = llm.invoke([HumanMessage(content=prompt)]).content
        
        import json
        import re
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Fallback recommendations
        return [
            {
                "type": "course",
                "title": f"Complete {target_role} certification",
                "description": "Industry-recognized certification will boost your profile",
                "priority": "high",
                "resource": "Check Coursera, Udemy, or official certification programs"
            }
        ]
    
    except Exception as e:
        return [{"error": str(e)}]


def compare_with_benchmarks(
    user_profile: Dict[str, Any],
    target_role: str
) -> Dict[str, Any]:
    """Compare user profile against industry benchmarks."""
    try:
        llm = get_llm()
        
        # Get industry benchmarks via web search
        from utils.web_search import web_search
        search_query = f"{target_role} industry average salary experience skills benchmark"
        web_results = web_search(search_query)
        
        benchmark_text = "\n".join([r.get("content", "")[:500] for r in web_results[:3]]) if web_results else ""
        
        prompt = f"""Compare this candidate against industry benchmarks for {target_role}.

CANDIDATE:
{str(user_profile)[:1500]}

INDUSTRY BENCHMARKS:
{benchmark_text}

Return JSON:
{{
  "experience_vs_benchmark": "above|at|below",
  "skills_vs_benchmark": "above|at|below",
  "education_vs_benchmark": "above|at|below",
  "overall_position": "top 10%|top 25%|average|below average",
  "benchmark_details": {{
    "avg_experience": "X years",
    "avg_skills_count": "X skills",
    "common_certifications": ["cert1", "cert2"]
  }}
}}
"""
        
        result = llm.invoke([HumanMessage(content=prompt)]).content
        
        import json
        import re
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        return {
            "experience_vs_benchmark": "at",
            "skills_vs_benchmark": "at",
            "education_vs_benchmark": "at",
            "overall_position": "average"
        }
    
    except Exception as e:
        return {"error": str(e)}

